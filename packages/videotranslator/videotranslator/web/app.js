import {copy, normalizeLocale} from './i18n.js?v=20260929-stage-progress';
import {createLanguagePicker} from './language-picker.js?v=20260923-stripe-env';
import {createAccount} from './account.js?v=20260924-admin-pricing';
import {createHistory, progressStageText, queueMessageKey} from './history.js?v=20260929-stage-progress';
import {authenticatedFetch} from './authenticated-request.js?v=20260923-stripe-env';
import {createErrorNotice} from './error-notice.js?v=20260923-stripe-env';
import {uploadBlob} from './blob-upload.js?v=20260923-azure';

import {createAdminPricing, pricingText} from './admin-pricing.js?v=20260924-pricing';

const $ = id => document.getElementById(id);
const languagePicker = createLanguagePicker($('locale'));
for (const option of $('locale').options) $('auth-locale').append(option.cloneNode(true));
const authLanguagePicker = createLanguagePicker($('auth-locale'));
let language = normalizeLocale(navigator.languages?.[0] || navigator.language);
let draft, config, selectedFile, tokenProvider, account, history, owner, verified = false, busy = false, timer, uploadPercent = null;
let restorePromise, adminPricing;
const saved = (key, value) => {try {if (value === undefined) return localStorage.getItem(key); if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);} catch {} return null;};
let preferredLocale = saved('vt.locale') || '';
if (preferredLocale) {
  preferredLocale = normalizeLocale(preferredLocale);
  saved('vt.locale', preferredLocale); language = preferredLocale;
}
const conversationKey = () => `vt.conversation.${config?.payment_mode || 'disabled'}.${owner}`;
const t = key => ['rateNotice','roundingNotice'].includes(key) ? pricingText(key, key === 'roundingNotice' && draft?.quote ? draft.quote : config, language) : copy[language]?.[key] || copy.en[key] || key;
const errorNotice = createErrorNotice({element:$('error'), t,
  isRunning:() => !!draft?.job && !['succeeded','failed','cancelled','expired'].includes(draft.job.status)});
const node = (tag, className, text) => {const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el;};

function localize() {
  document.documentElement.lang = language;
  document.querySelectorAll('[data-i18n]').forEach(el => el.textContent = t(el.dataset.i18n));
  $('message').placeholder = t('placeholder');
  $('send').ariaLabel = t('send'); $('attach').ariaLabel = t('attach');
  $('message').ariaLabel = t('placeholder');
  $('locale').options[0].textContent = t('auto');
  const explicit = draft ? draft.explicit_locale : preferredLocale;
  $('locale').value = explicit ? normalizeLocale(explicit) : 'auto';
  $('auth-locale').value = $('locale').value;
  $('locale').setAttribute('aria-label',t('interfaceLanguage'));
  $('auth-locale').setAttribute('aria-label',t('interfaceLanguage'));
  languagePicker.update(language, t('auto'));
  authLanguagePicker.update(language, t('auto'));
  account?.localize();
  adminPricing?.localize();
  $('payment-mode-badge').hidden = config?.payment_mode !== 'sandbox';
  history?.render();
}

async function api(path, body, method = body === undefined ? 'GET' : 'POST') {
  const requestOwner = owner;
  const response = await authenticatedFetch('/api/v1' + path, {
    method, headers: {'X-Payment-Mode': config?.payment_mode || 'disabled', ...(body === undefined ? {} : {'Content-Type': 'application/json'})},
    body: body === undefined ? undefined : JSON.stringify(body),
  }, {getToken: force => tokenProvider?.(force), isCurrent: () => owner === requestOwner});
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const code = data.detail?.code;
    const aliases = {stale_job_version:'draft_changed',invalid_transition:'draft_changed',no_audio_track:'media_inspection_failed',forbidden:'unauthenticated',not_found:'network'};
    const error = new Error(Object.hasOwn(copy.en, data.detail?.message || '') ? data.detail.message : aliases[code] || code || 'network');
    error.code = code || (response.status === 404 ? 'not_found' : undefined);
    throw error;
  }
  const data = await response.json();
  errorNotice.authenticated();
  return data;
}

function showError(error, options) {errorNotice.show(error, options);}
const canContinue = () => draft?.status === 'submitted' && ['succeeded','failed','cancelled','expired'].includes(draft.job?.status);
function setBusy(value) {
  busy = value;
  languagePicker.setDisabled(value);
  authLanguagePicker.setDisabled(value);
  $('busy-indicator')?.remove();
  if (value && draft) {const indicator = node('div', 'busy-dots', t('working')); indicator.id = 'busy-indicator'; indicator.role = 'status'; $('actions').append(indicator);}
  document.querySelectorAll('#actions button, #composer button, #locale, #restart').forEach(button => button.disabled = value || button.dataset.unavailable === 'true');
  $('restart').disabled = value || !verified;
  $('sidebar-new').disabled = value || !verified;
  $('signout').disabled = value;
  $('history-open').disabled = value || !verified;
  $('message').disabled = value || !draft || (draft.status !== 'draft' && !canContinue());
  $('attach').disabled = value || !draft || (!canContinue() && (!!draft.job || !['draft', 'uploading'].includes(draft.status)));
  $('send').disabled = $('message').disabled;
}
async function operation(fn) {
  if (busy) return;
  $('error').hidden = true; setBusy(true);
  try {await fn();} catch (error) {
    if (error.message === 'draft_changed' && draft) {
      try {draft = await api(`/conversations/${draft.conversation_id}`); render();} catch {}
    }
    showError(error);
  } finally {setBusy(false);}
}

function button(label, fn, className = 'chip') {
  const el = node('button', className, label); el.type = 'button'; el.addEventListener('click', () => operation(fn)); return el;
}
function addMessage(role, text) {
  const wrap = node('article', `message ${role}`);
  if (role === 'assistant') wrap.append(node('span', 'avatar', '✦'));
  const content = node('div', 'message-content');
  if (role === 'assistant') content.append(node('span', 'message-name', t('assistant')));
  content.append(document.createTextNode(text)); wrap.append(content); $('messages').append(wrap);
}
const promptFor = step => ({source:'welcome',link:'link',file:'file',target:'targetPrompt',review:'reviewPrompt'})[step] || step;
function userMessage(message) {
  if (message.text) return message.text;
  if (message.filename) return `${t('fileSelected')} · ${message.filename}`;
  if (message.choice === 'source') return t(message.value === 'youtube' ? 'youtube' : 'upload');
  if (message.choice === 'target') return `${t('targetLabel')} · ${t(message.value)}`;
  if (message.choice === 'edit_source') return t('newSource');
  if (message.choice === 'edit_target') return t('newTarget');
  if (message.choice === 'locale') return `${t('localeSet')} ${message.value === 'auto' ? t('auto') : message.value}`;
  return t('nothing');
}
async function edit(fields) {
  if (fields.choice !== 'locale' && canContinue()) await continueConversation();
  draft = await api(`/conversations/${draft.conversation_id}/messages`, {revision:draft.revision, ...fields});
  if (fields.choice === 'edit_source') selectedFile = undefined;
  render(true);
}

function render(scroll = false) {
  const hasMessages = !!draft?.messages.some(message => message.choice !== 'locale');
  document.body.dataset.session = owner ? 'signed-in' : 'guest';
  document.body.dataset.started = String(hasMessages);
  document.querySelector('.intro').hidden = hasMessages;
  $('restart').hidden = !hasMessages;
  if (!draft) {
    localize(); $('messages').replaceChildren(); $('actions').replaceChildren();
    if (!owner) $('actions').append(button(t('signIn'), async () => $('signin').click(), 'primary'));
    else if (verified) {
      if (restorePromise) $('actions').append(node('p','composer-hint',t('working')));
      else $('actions').append(button(t('restoreSession'), restoreConversation, 'primary'));
    }
    setBusy(busy); return;
  }
  language = normalizeLocale(draft.locale); localize();
  $('demo').hidden = !config.demo;
  $('messages').replaceChildren(); $('actions').replaceChildren();
  for (const message of draft.messages) {
    if (message.choice === 'locale') continue;
    if (message.job_id) {
      addMessage('assistant', `${t('previousTask')} · ${message.filename || message.job_id}`);
      $('messages').lastElementChild.querySelector('.message-content').append(button(t('viewTask'), async () => history.open(message.job_id), 'text-button'));
      continue;
    }
    addMessage(message.role, message.role === 'user' ? userMessage(message) : (message.text || t(promptFor(message.step))));
  }
  const actions = $('actions');
  if (draft.status === 'draft') {
    if (draft.source_kind && draft.step !== 'review') {
      const navigation = node('div', 'source-navigation');
      navigation.append(button(`← ${t('backToSources')}`, () => edit({choice:'edit_source'}), 'text-button'));
      actions.append(navigation);
    }
    if (draft.step === 'source') {
      const grid = node('div', 'choice-grid');
      for (const [kind, icon, title, detail] of [['youtube','▶','youtube','youtubeDetail'],['upload','↥','upload','uploadDetail']]) {
        const unavailable = kind === 'youtube' && config.youtube_available === false;
        const card = button('', async () => {await edit({choice:'source',value:kind}); if (kind === 'upload') $('file').click();}, 'choice-card');
        card.dataset.unavailable = String(unavailable); card.disabled = unavailable;
        if (unavailable) card.setAttribute('aria-disabled', 'true');
        card.append(node('span','choice-icon',icon));
        const text = node('span'); text.append(node('strong','',t(title)),node('small','',t(unavailable ? 'youtubeUnavailable' : detail))); card.append(text,node('span','arrow',unavailable ? '—' : '↗')); grid.append(card);
      }
      actions.append(grid);
    } else if (draft.step === 'file') {
      const choices = node('div','chips'); choices.append(button(t('selectFile'), async () => $('file').click())); actions.append(choices);
    } else if (draft.step === 'target') {
      const choices = node('div','chips');
      for (const code of config.target_languages) choices.append(button(t(code), () => edit({choice:'target',value:code})));
      actions.append(choices);
    } else if (draft.step === 'review') renderReview(actions);
    if (draft.interpreter_mode === 'fallback') actions.append(node('p','notice',t('fallback')));
  } else renderJob(actions);
  setBusy(busy);
  if (scroll && $('history-view').hidden) requestAnimationFrame(() => $('scroll-area').scrollTop = $('scroll-area').scrollHeight);
  schedulePoll();
}

function renderReview(parent) {
  const card = node('div','review-card'); const heading = node('div','card-heading',t('confirmTitle'));
  heading.append(node('span','review-badge',t('ready'))); card.append(heading);
  for (const [label,value,choice] of [['sourceLabel',draft.source_kind === 'youtube' ? draft.youtube_url : `${draft.filename} · ${(draft.size_bytes / 1024 / 1024).toFixed(1)} MB`,'edit_source'],['targetLabel',t(draft.target_language),'edit_target']]) {
    const row = node('div','review-row'), text = node('div'); text.append(node('small','',t(label)), node('strong','',value));
    const change = button(t('edit'), () => edit({choice})); change.ariaLabel = t(choice === 'edit_source' ? 'newSource' : 'newTarget');
    row.append(text,change); card.append(row);
  }
  const footer = node('div','card-footer'); card.append(footer); parent.append(card);
  if (draft.quote) {
    const quote = draft.quote, details = node('div','quote-details');
    const line = (label, value, extra = '') => {const row = node('div',`quote-line ${extra}`); row.append(node('span','',t(label)),node('strong','',value)); details.append(row);};
    line('duration', `${(quote.duration_ms / 60000).toFixed(2)} ${t('minutes')}`);
    line('unitPrice', `${account.money(quote.rate_cents_per_minute)} / ${t('minute')}`);
    line('totalCost', `${account.money(quote.amount_cents)}`, 'total');
    if (account.balance !== null) {
      line('currentBalance', `${account.money(account.balance)}`);
      if (account.balance >= quote.amount_cents) line('remainingBalance', `${account.money(account.balance - quote.amount_cents)}`);
      else details.append(node('p','low-balance',t('insufficient_credits')));
    }
    details.append(node('p','',t('roundingNotice'))); card.insertBefore(details,footer);
    if (config.processing_available === false) footer.append(node('p','notice',t('processing_unavailable')));
    else if (account.balance !== null && account.balance < quote.amount_cents) footer.append(button(t('topup'), account.openTopup, 'primary'));
    else {const start = button(`${t('confirm')} · ${account.money(quote.amount_cents)}`, confirm, 'primary'); start.dataset.unavailable = String(account.balance === null); footer.append(start);}
    footer.append(node('p','',t('confirmHint')));
    footer.append(node('p','',t('failureCreditPolicy')));
  } else {
    footer.append(button(t('prepareQuote'), prepare, 'primary'),node('p','',t('prepareHint')));
  }
}

function renderJob(parent) {
  const job = draft.job, status = job?.stage === 'queued' && job?.status === 'provisioning' ? 'queued' : job?.status || draft.status;
  const card = node('div','job-card'); card.append(node('div','card-heading',t('confirmTitle')));
  const body = node('div','job-body'); body.append(node('div','',t(status === 'awaiting_capacity' ? queueMessageKey(job) : status)));
  if (!job && draft.download?.status === 'queued') {
    const attempts = draft.download.unavailable_retries;
    body.append(node('p', 'notice', attempts ? `${t('downloadReconnecting')} (${attempts}/10)` : t('downloadQueued')));
  }
  if (!job && draft.status === 'import_failed' && draft.error === 'youtube_proxy_unavailable') {
    body.append(node('p', 'notice', t('downloadUnavailable')));
  }
  if (job?.error_code === 'processing_unavailable' && config.processing_available === false) body.append(node('p','notice',t('processing_unavailable')));
  for (const warning of job?.warnings || []) body.append(node('p', 'notice', warning));
  if (job) {
    body.append(node('div','job-id',job.job_id));
    if (job.refund_status === 'completed') {
      const amount = job.balance_returned_cents ? ` · ${account.money(job.balance_returned_cents)}` : '';
      body.append(node('p','notice',`${t('balanceReturned')}${amount}\n${t('creditReusable')}`));
    } else if (status === 'failed') body.append(node('p','notice',t(job.charged_ledger_entry_id ? 'creditPending' : 'noTaskDebit')));
    if (['running','provisioning','submitting'].includes(status)) {
      const progress = node('progress'); progress.max = 100; progress.value = Math.max(0,Math.min(100,job.progress_percent || 0)); progress.ariaLabel = t('progress');
      body.append(progress,node('small','job-stage',progressStageText(job,t)));
    }
    const choices = node('div','secondary-actions');
    if (status === 'awaiting_credits') choices.append(button(t('topup'), account.openTopup));
    if (config.processing_available !== false && status === 'awaiting_credits') choices.append(button(t('start'), () => reviewJobAction('start'), 'primary'));
    if (['awaiting_capacity','queued','submitting','provisioning','running'].includes(status)) choices.append(button(t('cancel'), () => jobAction('cancel')));
    if (status === 'succeeded') {
      choices.append(button(t('download'), downloadResult, 'primary'));
      choices.append(button(t('previewResult'), () => previewResult(job, body)));
    }
    if (config.processing_available !== false && status === 'failed' && (job.retry_allowed || job.error_code === 'processing_unavailable')) choices.append(button(t('retry'), () => reviewJobAction('retry'), 'primary'));
    body.append(choices);
    if (canContinue()) {
      body.append(node('p','continue-hint',t('continueHint')),
        button(t('nextTranslation'), continueConversation, 'chip'));
    }
  } else if (draft.status === 'uploading') {
    if (uploadPercent !== null) {const progress = node('progress'); progress.max = 100; progress.value = uploadPercent; progress.ariaLabel = t('progress'); body.append(progress); body.append(node('span','',`${uploadPercent}%`));}
    body.append(button(t('resumeUpload'), resumeUpload, 'primary'));
  } else if (draft.status === 'import_failed') body.append(button(t('retryImport'), prepare, 'primary'));
  else if (draft.status === 'confirming') body.append(button(t('resumeConfirmation'), confirm, 'primary'));
  card.append(body); parent.append(card);
}

function sameFile() {return selectedFile && selectedFile.name === draft.filename && selectedFile.size === draft.size_bytes;}
async function prepare() {
  if (draft.source_kind === 'upload' && !sameFile()) {showError(new Error('reselect')); $('file').click(); return;}
  draft = await api(`/conversations/${draft.conversation_id}/prepare`, {revision:draft.revision}); render(true);
  if (draft.status === 'uploading' && !draft.job) await resumeUpload();
}
async function confirm() {
  draft = await api(`/conversations/${draft.conversation_id}/confirm`, {revision:draft.revision});
  await account.refresh(); history.refresh(); render(true);
}

async function resumeUpload() {
  draft = await api(`/conversations/${draft.conversation_id}`);
  if (draft.job) {render(); return;}
  if (!sameFile()) {showError(new Error('reselect')); $('file').click(); return;}
  try {
    const upload = await api(`/uploads/${draft.upload_id}/renew`, {});
    const token = await tokenProvider(); uploadPercent = 0;
    const url = config.local ? `/api/v1/uploads/${upload.upload_id}/content` : upload.upload_url;
    if (!config.local) {
      await uploadBlob(selectedFile, url, {
        renew: () => api(`/uploads/${upload.upload_id}/renew`, {}),
        onProgress: (loaded, total) => {uploadPercent = Math.round(loaded * 100 / total); $('actions').replaceChildren(); renderJob($('actions')); setBusy(true);}
      });
    } else await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); xhr.open('PUT', url);
      if (config.local) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      else xhr.setRequestHeader('x-ms-blob-type', 'BlockBlob');
      xhr.setRequestHeader('Content-Type', selectedFile.type || 'application/octet-stream');
      xhr.upload.onprogress = event => {if (event.lengthComputable) {uploadPercent = Math.round(event.loaded * 100 / event.total); $('actions').replaceChildren(); renderJob($('actions')); setBusy(true);}};
      xhr.onload = () => xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error('uploadFailed'));
      xhr.onerror = xhr.onabort = () => reject(new Error('uploadFailed'));
      xhr.send(selectedFile);
    });
    draft = await api(`/conversations/${draft.conversation_id}/inspect`, {}); render(true);
  } finally {uploadPercent = null;}
}

async function reviewJobAction(action) {
  clearTimeout(timer);
  const amount = draft.job.quoted_point_units;
  $('actions').replaceChildren();
  const card = node('div','review-card'), body = node('div','job-body');
  body.append(node('p','',`${t('totalCost')}: ${account.money(amount)}`),node('p','',t('confirmHint')));
  body.append(button(`${t('confirm')} · ${account.money(amount)}`, () => jobAction(action), 'primary'),button(t('back'), async () => render()));
  card.append(body); $('actions').append(card);
}
async function jobAction(action) {
  const response = await api(`/jobs/${draft.job.job_id}/${action}`, {});
  if (action === 'retry') {
    // Retries are independent jobs; the conversation follows the returned retry id.
    await api(`/conversations/${draft.conversation_id}/follow-retry`, {job_id:response.job_id});
  }
  draft = await api(`/conversations/${draft.conversation_id}`); await account.refresh(); history.refresh(); render();
}
async function downloadResult() {
  return downloadJob(draft.job);
}
async function previewResult(job, parent) {
  const result = await api(`/jobs/${job.job_id}/result`);
  if (!parent.isConnected || parent.closest('[hidden]')) return;
  parent.querySelector('video, audio')?.remove();
  const player = node(job.media_type === 'audio' ? 'audio' : 'video');
  player.controls = true; player.preload = 'metadata'; player.style.width = '100%';
  player.crossOrigin = 'anonymous'; player.playsInline = true;
  player.src = result.download_url; player.ariaLabel = t('previewResult');
  if (job.media_type !== 'audio' && result.subtitle_url) {
    const track = node('track');
    track.kind = 'subtitles'; track.src = result.subtitle_url;
    track.srclang = result.subtitle_language || job.target_language;
    track.label = track.srclang === 'zh' ? '中文' : 'English';
    track.default = true;
    player.append(track);
  }
  parent.append(player);
  await player.play().catch(() => {});
}
async function downloadJob(job) {
  const result = await api(`/jobs/${job.job_id}/result`);
  if (result.download_url.startsWith('/api/v1/jobs/')) {
    const link = node('a'); link.href = result.download_url + '&download=true';
    link.download = job.media_type === 'audio' ? 'translated.mp3' : 'translated.mp4'; link.click(); return;
  }
  if (!config.local) {const link = node('a'); link.href = result.download_url; link.rel = 'noopener'; link.target = '_blank'; link.click(); return;}
  const response = await fetch(`/api/v1/jobs/${job.job_id}/content`, {headers:{Authorization:`Bearer ${await tokenProvider()}`}});
  if (!response.ok) throw new Error('resultError');
  const url = URL.createObjectURL(await response.blob()), link = node('a'); link.href = url; link.download = job.media_type === 'audio' ? 'translated.mp3' : 'translated.mp4'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);
}
function schedulePoll() {
  clearTimeout(timer);
  if (!draft || draft.status === 'draft' || ['succeeded','failed','cancelled','expired'].includes(draft.job?.status)) return;
  timer = setTimeout(async () => {
    if (!busy) try {
      const returnedBefore = draft.job?.refund_status;
      const chargedBefore = draft.job?.charged_ledger_entry_id;
      draft = await api(`/conversations/${draft.conversation_id}`);
      errorNotice.recovered('poll');
      if ((draft.job?.refund_status === 'completed' && returnedBefore !== 'completed') ||
          (draft.job?.charged_ledger_entry_id && draft.job.charged_ledger_entry_id !== chargedBefore)) await account.refresh();
      render();
    } catch (error) {showError(error, {background:true, source:'poll'});}
    schedulePoll();
  }, 3000);
}

async function newConversation() {
  history.close();
  selectedFile = undefined; uploadPercent = null;
  draft = await api('/conversations', {locale:preferredLocale || navigator.languages?.[0] || navigator.language || 'en'});
  if (preferredLocale) draft = await api(`/conversations/${draft.conversation_id}/messages`, {revision:draft.revision,choice:'locale',value:preferredLocale});
  saved(conversationKey(), draft.conversation_id); $('message').value = ''; render(true);
}
async function continueConversation() {
  draft = await api(`/conversations/${draft.conversation_id}/continue`, {revision:draft.revision});
  selectedFile = undefined; uploadPercent = null; render(true);
}
$('sidebar-new').addEventListener('click', () => history.close());
$('restart').addEventListener('click', () => operation(newConversation));
function changeLocale(event) {
  const value = event.target.value;
  operation(async () => {
    preferredLocale = value === 'auto' ? '' : value; saved('vt.locale',preferredLocale);
    if (draft) await edit({choice:'locale',value});
    else {language = normalizeLocale(preferredLocale || navigator.languages?.[0] || navigator.language); render();}
  });
}
$('locale').addEventListener('change', changeLocale);
$('auth-locale').addEventListener('change', changeLocale);
$('attach').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', event => {
  const file = event.target.files[0]; event.target.value = ''; if (!file) return;
  operation(async () => {
    if (!draft) return;
    if (draft.status === 'uploading') {
      if (file.name !== draft.filename || file.size !== draft.size_bytes) throw new Error('wrongFile');
      selectedFile = file; await resumeUpload();
    } else {
      await edit({filename:file.name,size_bytes:file.size}); selectedFile = file;
    }
  });
});
$('composer').addEventListener('submit', event => {
  event.preventDefault(); const text = $('message').value.trim();
  if (!text || !draft || (draft.status !== 'draft' && !canContinue())) return;
  operation(async () => {await edit({text}); $('message').value = ''; $('message').style.height = '';});
});
$('message').addEventListener('keydown', event => {if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); $('composer').requestSubmit();}});
$('message').addEventListener('input', () => {$('message').style.height = 'auto'; $('message').style.height = Math.min($('message').scrollHeight, 140) + 'px';});

async function restoreConversation() {
  if (restorePromise) return restorePromise;
  restorePromise = Promise.resolve().then(async () => {
    render();
    const key = saved(conversationKey());
    if (key) {
      try {draft = await api(`/conversations/${encodeURIComponent(key)}`);}
      catch (error) {
        if (!['conversation_not_found','not_found','forbidden'].includes(error.code || error.message)) throw error;
        saved(conversationKey(),null);
      }
    }
    if (draft) {
      if (preferredLocale && preferredLocale !== draft.explicit_locale) await edit({choice:'locale',value:preferredLocale});
      else render();
    } else await newConversation();
    $('error').hidden = true;
  }).finally(() => {restorePromise = undefined; render();});
  return restorePromise;
}

async function boot() {
  localize(); setBusy(true);
  try {
    config = await fetch('/api/v1/chat/config').then(response => response.json());
    $('demo').hidden = !config.demo;
    adminPricing = createAdminPricing({api, language:() => language, updated:price => {Object.assign(config, price); render();}});
    account = await createAccount({config,api,t,report:error => showError(error, {background:true, source:'balance'}),changed:async user => {
      if (owner === user?.uid && verified === !!user?.emailVerified && (!verified || draft)) return;
      clearTimeout(timer); draft = undefined; selectedFile = undefined; uploadPercent = null;
      owner = user?.uid; verified = !!user?.emailVerified;
      history.setUser(verified);
      $('error').hidden = true;
      render();
      if (!verified) {render(); return;}
      await restoreConversation();
    }});
    tokenProvider = force => account.getToken(force);
    await account.start(); render();
  } catch (error) {showError(error);} finally {setBusy(false);}
}
document.addEventListener('balance-updated', () => {errorNotice.recovered('balance'); if (draft?.status === 'draft' && draft.quote) render();});
history = createHistory({api,t,money:cents => account.money(cents),download:downloadJob,preview:previewResult,refreshBalance:() => account.refresh()});
boot();

async function refreshYouTubeAvailability() {
  if (!config || document.hidden) return;
  try {
    const response = await fetch('/api/v1/chat/youtube-availability', {cache:'no-store'});
    if (!response.ok) return;
    const latest = await response.json();
    const availabilityChanged = latest.available !== config.youtube_available;
    config.youtube_available = latest.available;
    if (availabilityChanged && draft?.status === 'draft') render();
  } catch { /* Keep the last known presence; server validation remains authoritative. */ }
}
async function refreshPublicPricing() {
  if (!config || document.hidden) return;
  try {
    const response = await fetch('/api/v1/chat/config', {cache:'no-store'});
    if (!response.ok) return;
    const latest = await response.json();
    if (latest.pricing_version !== config.pricing_version) {Object.assign(config, latest); localize();}
  } catch { /* Keep the last known price; confirmation uses the server quote. */ }
}
window.addEventListener('focus', () => {refreshYouTubeAvailability(); refreshPublicPricing();});
setInterval(refreshYouTubeAvailability, 5000);
setInterval(refreshPublicPricing, 60000);
