export function statusGroup(status, stage = '') {
  if (status === 'provisioning' && stage === 'queued') return 'queued';
  if (['queued', 'awaiting_capacity'].includes(status)) return 'queued';
  if (['inspecting','submitting','provisioning','running','cancelling'].includes(status)) return 'running';
  if (['succeeded','failed','cancelled','expired'].includes(status)) return status;
  return 'waiting';
}
const statusKeys = {queued:'statusQueued',running:'statusRunning',failed:'statusFailed',succeeded:'statusSucceeded',waiting:'statusWaiting',cancelled:'statusCancelled',expired:'statusExpired'};

const stageKeys = {
  import:'stageImport', queued:'stageQueued', starting_gpu:'stageStartingGpu',
  prepare:'stagePrepare', extract:'stageExtract', separate:'stageSeparate',
  transcribe:'stageTranscribe', segment:'stageSegment', translate:'stageTranslate',
  synthesize:'stageSynthesize', mix:'stageMix', assemble:'stageAssemble',
  upload:'stageUpload', complete:'stageComplete', done:'stageComplete',
};

export function progressStageText(job, t) {
  const fallback = {
    inspecting:'stageInspecting', submitting:'stageSubmitting',
    provisioning:'stageProvisioning', cancelling:'stageCancelling',
  }[job?.status] || 'stageProcessing';
  const stage = t(stageKeys[job?.stage] || fallback);
  const percent = Math.max(0, Math.min(100, Number(job?.progress_percent) || 0));
  return `${t('currentStage')}: ${stage} · ${percent}%`;
}

export function queueMessageKey(job) {
  return ['gpu_starts_disabled', 'cost_budget_exceeded'].includes(job?.capacity_wait_reason)
    ? 'translationPaused' : 'awaiting_capacity';
}

export function createHistory({api, t, money, download, preview, refreshBalance}) {
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => {const item = document.createElement(tag); item.className = cls; if (text !== undefined) item.textContent = text; return item;};
  let jobs = [], selected = '', generation = 0, loading = false, allowed = false, poll, chatScroll = 0, listScroll = 0, detailSignature = '';
  const available = job => job.status === 'succeeded' && !job.assets_deleted_at && job.output_object_key && (!job.output_expires_at || job.output_expires_at > Date.now());
  const date = value => value ? new Intl.DateTimeFormat(document.documentElement.lang,{dateStyle:'medium',timeStyle:'short'}).format(value) : '—';
  function badge(job) {const group = statusGroup(job.status, job.stage); return el('span',`task-badge ${group}`,t(statusKeys[group]));}
  function render() {
    syncNavigation();
    renderSidebar();
    if ($('history-view').hidden) return;
    const filtered = jobs.filter(job => $('history-filter').value === 'all' || statusGroup(job.status, job.stage) === $('history-filter').value);
    $('history-count').textContent = `${filtered.length} / ${jobs.length}`;
    $('history-empty').hidden = !!filtered.length;
    $('history-empty').textContent = t(loading && !jobs.length ? 'working' : jobs.length ? 'historyNoMatch' : 'historyEmpty');
    const focusId = document.activeElement?.dataset.jobId, focusAction = document.activeElement?.dataset.action;
    $('history-list').replaceChildren();
    for (const job of filtered) {
      const row = el('article','history-item');
      const name = el('button','history-task-name'); name.type = 'button'; name.dataset.jobId = job.job_id; name.dataset.action = 'open';
      name.append(el('strong','',job.original_filename || job.job_id));
      const duration = job.duration_ms ? `${(job.duration_ms/60000).toFixed(2)} ${t('minutes')}` : '—';
      name.append(el('span','task-meta',`${t(job.media_type === 'audio' ? 'mediaAudio' : 'mediaVideo')} → ${t(job.target_language)} · ${duration} · ${money(job.quoted_point_units)}`));
      name.onclick = () => showDetail(job.job_id);
      const state = el('div','history-task-state'); state.append(badge(job));
      if (job.refund_status === 'completed') state.append(el('small','',t('balanceReturned')));
      if (statusGroup(job.status,job.stage) === 'running') state.append(el('small','',`${job.progress_percent || 0}%`));
      const actions = el('div','history-task-actions');
      const detailButton = el('button','text-button',t('viewTask')); detailButton.type = 'button'; detailButton.dataset.jobId = job.job_id; detailButton.dataset.action = 'detail';
      detailButton.onclick = () => showDetail(job.job_id); actions.append(detailButton);
      if (available(job)) {
        const link = el('button','chip',t('download')); link.type = 'button'; link.dataset.jobId = job.job_id; link.dataset.action = 'download';
        link.onclick = async () => {link.disabled = true; try {await download(job);} catch {showError('resultError');} finally {link.disabled = false;}};
        actions.append(link);
      }
      row.append(name,state,el('time','history-task-date',date(job.created_at)),actions); $('history-list').append(row);
      for (const control of row.querySelectorAll('button')) if (control.dataset.jobId === focusId && control.dataset.action === focusAction) control.focus({preventScroll:true});
    }
    const job = filtered.find(item => item.job_id === selected);
    const signature = JSON.stringify([job,document.documentElement.lang]);
    if (detailSignature === signature) return; // Keep an active player while other tasks update.
    detailSignature = signature;
    const detail = $('history-detail'); detail.hidden = !job; detail.replaceChildren();
    if (!job) return;
    detail.append(el('h2','',job.original_filename || job.job_id),badge(job));
    const line = (key, value) => {const row = el('div','history-detail-row'); row.append(el('span','',t(key)),el('strong','',value)); detail.append(row);};
    line('taskId',job.job_id); line('targetLabel',t(job.target_language));
    line('duration',job.duration_ms ? `${(job.duration_ms / 60000).toFixed(2)} ${t('minutes')}` : '—');
    line('totalCost', `${money(job.quoted_point_units)}`);
    if (job.refund_status === 'completed') {
      line('balanceReturned',job.balance_returned_cents ? `${money(job.balance_returned_cents)}` : t('balanceReturned'));
      if (job.balance_returned_at) line('balanceReturnedTime',date(job.balance_returned_at));
      detail.append(el('p','notice',t('creditReusable')));
    } else if (job.status === 'failed') detail.append(el('p','notice',t(job.charged_ledger_entry_id ? 'creditPending' : 'noTaskDebit')));
    line('createdTime',date(job.created_at));
    if (job.queued_at) line('queuedTime',date(job.queued_at));
    if (job.started_at) line('startedTime',date(job.started_at));
    if (job.completed_at) line('completedTime',date(job.completed_at));
    if (statusGroup(job.status, job.stage) === 'running') {
      const progress = el('progress',''); progress.max = 100; progress.value = Math.max(0,Math.min(100,job.progress_percent || 0)); progress.ariaLabel = t('progress');
      detail.append(progress,el('small','history-stage',progressStageText(job,t)));
    }
    if (job.status === 'failed') detail.append(el('p','notice',t('failed')));
    for (const warning of job.warnings || []) detail.append(el('p','notice',warning));
    if (['awaiting_credits','awaiting_capacity'].includes(job.status)) detail.append(el('p','notice',t(job.status === 'awaiting_capacity' ? queueMessageKey(job) : job.status)));
    if (job.retry_of_job_id) line('originalTask',job.retry_of_job_id);
    if (job.status === 'succeeded') {
      if (job.assets_deleted_at || !job.output_object_key || (job.output_expires_at && job.output_expires_at <= Date.now())) detail.append(el('p','notice',t('resultUnavailable')));
      else {
        const button = el('button','primary',t('download')); button.type = 'button';
        button.onclick = async () => {button.disabled = true; try {await download(job);} catch {showError('resultError');} finally {button.disabled = false;}};
        detail.append(button);
        if (preview) {
          const play = el('button','chip',t('previewResult')); play.type = 'button';
          play.onclick = async () => {play.disabled = true; try {await preview(job,detail);} catch {showError('resultError');} finally {play.disabled = false;}};
          detail.append(play);
        }
      }
    }
  }
  function syncNavigation() {
    const active = !$('history-view').hidden;
    document.body.dataset.view = active ? (selected ? 'detail' : 'history') : 'conversation';
    for (const [id,current] of [['sidebar-new',!active],['history-open',active]]) {
      if (current) $(id).setAttribute('aria-current','page'); else $(id).removeAttribute('aria-current');
    }
    const label = active ? (selected ? 'taskDetails' : 'historyTitle') : 'assistant';
    $('page-label').dataset.i18n = label; $('page-label').textContent = t(label);
    $('history-title').dataset.i18n = selected ? 'taskDetails' : 'historyTitle';
    $('history-title').textContent = t(selected ? 'taskDetails' : 'historyTitle');
    $('history-list-panel').hidden = !!selected;
    $('history-back-list').hidden = !selected;
  }
  function stopPlayback() {
    for (const player of $('history-detail').querySelectorAll('video,audio')) {player.pause(); player.remove();}
  }
  function showDetail(jobId) {
    listScroll = $('scroll-area').scrollTop;
    stopPlayback(); selected = jobId; render(); $('scroll-area').scrollTop = 0;
  }
  function renderSidebar() {
    $('sidebar-task-count').textContent = String(jobs.length);
    $('sidebar-task-count').hidden = !jobs.length;
  }
  function showError(key) {
    $('history-error').textContent = t(key); $('history-error').hidden = false;
  }
  async function refresh() {
    if (loading || !allowed) return;
    const requestGeneration = generation; loading = true; $('history-refresh').disabled = true;
    let changed = false;
    if (!jobs.length) render();
    try {
      const result = await api('/jobs');
      if (generation !== requestGeneration) return;
      const updated = result.jobs.sort((a,b) => b.created_at - a.created_at || b.job_id.localeCompare(a.job_id));
      const returned = updated.some(job => job.refund_status === 'completed' && !jobs.some(old => old.job_id === job.job_id && old.refund_status === 'completed'));
      changed = JSON.stringify(updated) !== JSON.stringify(jobs); jobs = updated;
      $('history-error').hidden = true;
      if (returned) await refreshBalance();
    } catch {if (generation === requestGeneration) showError('network');}
    finally {if (generation === requestGeneration) {loading = false; $('history-refresh').disabled = false; if (changed || !jobs.length) render();}}
  }
  function close() {
    const wasOpen = !$('history-view').hidden;
    stopPlayback();
    $('history-view').hidden = true; document.querySelector('.conversation').hidden = false;
    document.querySelector('.composer-wrap').hidden = false;
    if (wasOpen) $('scroll-area').scrollTop = chatScroll;
    syncNavigation();
  }
  function open(jobId) {
    if (!allowed) return;
    stopPlayback();
    selected = typeof jobId === 'string' ? jobId : '';
    $('history-filter').value = 'all'; listScroll = 0;
    if ($('history-view').hidden) chatScroll = $('scroll-area').scrollTop;
    $('history-view').hidden = false; document.querySelector('.conversation').hidden = true;
    document.querySelector('.composer-wrap').hidden = true; $('scroll-area').scrollTop = 0;
    render(); refresh();
  }
  $('history-open').onclick = () => open(); $('history-close').onclick = close;
  $('history-back-list').onclick = () => {stopPlayback(); selected = ''; render(); $('scroll-area').scrollTop = listScroll;};
  $('history-refresh').onclick = refresh; $('history-filter').onchange = () => {selected = ''; render();};
  return {close,open,render,refresh,setUser(signedIn) {
    generation++; loading = false; clearInterval(poll); close(); jobs = []; selected = ''; detailSignature = ''; allowed = signedIn;
    $('history-open').disabled = $('sidebar-new').disabled = !signedIn;
    $('history-list').replaceChildren(); $('history-detail').replaceChildren(); $('history-filter').value = 'all';
    $('history-error').hidden = true; render();
    if (allowed) {refresh(); poll = setInterval(() => {if (!document.hidden) refresh();},5000);}
  }};
}
