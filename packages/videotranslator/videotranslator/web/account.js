// Firebase owns credentials and token refresh. Money always comes from the API.
import {verifyReturnedPayment} from './payment-verification.js?v=20260923-payment';
import {packageLabel} from './topup-packages.js?v=20260923-stripe-env';
export async function createAccount({config, api, t, changed, report}) {
  const $ = id => document.getElementById(id);
  let user = null, sdk, auth, balance = null, lastRefresh = 0, generation = 0, paymentBusy = false;
  let authErrorKeys = [];
  let paymentState = 'choose', paymentMessage = '', lastPaymentCheck = 0;
  let topupPackages = [];
  const money = cents => new Intl.NumberFormat(document.documentElement.lang, {style:'currency', currency:'USD', currencyDisplay:'code'}).format(cents / 100);
  const sessionKey = () => `vt.checkout.${config.payment_mode}.${user?.uid}`;
  function paymentView(state, message = '') {
    paymentState = state; paymentMessage = message;
    renderPayment();
  }
  function renderPayment() {
    $('topup-form').hidden = paymentState !== 'choose';
    $('topup-title').textContent = t(paymentState === 'choose' ? 'topupTitle' : paymentState === 'succeeded' ? 'paymentSuccessTitle' : 'paymentVerifyTitle');
    $('payment-status').textContent = paymentMessage ? t(paymentMessage) : '';
    $('payment-status').setAttribute('aria-busy', String(paymentState === 'verifying'));
    $('payment-help').hidden = paymentState === 'choose' || paymentState === 'succeeded';
    $('payment-help').textContent = t('paymentNoRepeat');
    $('payment-retry').hidden = paymentState !== 'unconfirmed';
    $('payment-done').hidden = paymentState !== 'succeeded';
  }
  function localize() {
    if (auth) auth.languageCode = document.documentElement.lang === 'zh' ? 'zh-CN' : document.documentElement.lang;
    $('auth-error').textContent = authErrorKeys.map(t).join(' ');
    $('balance-label').textContent = t(config.auth_mode === 'demo' ? 'demoBalance' : 'balance');
    $('balance-value').textContent = balance === null ? '—' : `${money(balance)}`;
    $('topup-balance').textContent = $('balance-value').textContent;
    $('account-name').textContent = user?.email || t('guest');
    $('account-menu').hidden = !user;
    $('signin').hidden = !!user; $('signout').hidden = !user || config.auth_mode === 'demo';
    $('topup').disabled = !user?.emailVerified || config.auth_mode === 'demo';
    $('verify-panel').hidden = !user || user.emailVerified;
    $('auth-unconfigured').hidden = config.auth_mode === 'demo' || !!auth;
    $('google-signin').disabled = !auth; $('email-submit').disabled = !auth;
    $('reset-password').disabled = !auth;
    renderPayment();
    renderPackages();
  }
  function renderPackages() {
    if (!topupPackages.length) return;
    const select = $('topup-package'), selected = select.value;
    select.replaceChildren();
    for (const p of topupPackages) {
      const option = document.createElement('option'); option.value = p.package_id;
      option.textContent = packageLabel(p, money, t); select.append(option);
    }
    if (topupPackages.some(p => p.package_id === selected)) select.value = selected;
  }
  async function refresh() {
    if (!user) return;
    const currentGeneration = generation;
    try {
      const me = await api('/me');
      if (currentGeneration !== generation) return;
      balance = me.balance_cents; lastRefresh = Date.now(); localize();
      document.dispatchEvent(new Event('balance-updated'));
    } catch (error) {if (currentGeneration === generation) {balance = null; localize(); report(error);}}
  }
  async function update(next) {
    if (user?.uid !== next?.uid) {paymentView('choose'); $('topup-dialog').close();}
    generation++; user = next; balance = null; localize();
    try {await changed(next);}
    finally {if (next) {await refresh(); if (next.emailVerified) await paymentReturn();}}
  }
  async function authAction(fn) {
    authErrorKeys = []; localize();
    $('auth-error').textContent = ''; $('auth-notice').textContent = '';
    delete $('auth-notice').dataset.i18n;
    try {await fn();} catch (error) {
      const known = {'auth/invalid-credential':'authInvalid','auth/email-already-in-use':'authExists','auth/weak-password':'authWeak','auth/popup-closed-by-user':'authClosed','auth/too-many-requests':'authRate','auth/invalid-email':'authInvalidEmail'};
      const popupUnavailable = ['auth/popup-blocked','auth/operation-not-supported-in-this-environment','auth/web-storage-unsupported'].includes(error.code);
      const message = popupUnavailable ? 'authBrowserHelp' : known[error.code] || 'authFailed';
      if ($('auth-dialog').open) {authErrorKeys = [message, ...(error.code === 'auth/popup-closed-by-user' ? ['authBrowserHelp'] : [])]; localize();}
      else report(new Error(message));
    }
  }
  $('signin').onclick = () => {$('auth-dialog').showModal(); localize();};
  $('signout').onclick = () => authAction(async () => {if (auth) await sdk.signOut(auth);});
  document.addEventListener('click', event => {
    if (!$('account-menu').contains(event.target)) $('account-menu').open = false;
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && $('account-menu').open) {
      $('account-menu').open = false; $('account-menu').querySelector('summary').focus();
    }
  });
  $('google-signin').onclick = () => authAction(async () => {await sdk.signInWithPopup(auth, new sdk.GoogleAuthProvider()); $('auth-dialog').close();});
  $('email-form').onsubmit = event => {
    event.preventDefault();
    authAction(async () => {
      const email = $('auth-email').value.trim(), password = $('auth-password').value;
      const register = $('auth-register').checked;
      const result = await (register ? sdk.createUserWithEmailAndPassword : sdk.signInWithEmailAndPassword)(auth, email, password);
      $('auth-password').value = '';
      if (register) {await sdk.sendEmailVerification(result.user); $('verification-status').dataset.i18n = 'verifySent'; $('verification-status').textContent = t('verifySent');}
      $('auth-dialog').close();
    });
  };
  $('reset-password').onclick = () => authAction(async () => {
    if (!$('auth-email').value.trim()) {$('auth-email').focus(); return;}
    await sdk.sendPasswordResetEmail(auth, $('auth-email').value.trim()); $('auth-notice').dataset.i18n = 'resetSent'; $('auth-notice').textContent = t('resetSent');
  });
  $('resend-verification').onclick = () => authAction(async () => {await sdk.sendEmailVerification(auth.currentUser); $('verification-status').dataset.i18n = 'verifySent'; $('verification-status').textContent = t('verifySent');});
  $('check-verification').onclick = () => authAction(async () => {await sdk.reload(auth.currentUser); await auth.currentUser.getIdToken(true); await update(auth.currentUser);});
  document.querySelectorAll('[data-close-dialog]').forEach(button => button.onclick = () => $(button.dataset.closeDialog).close());
  async function openTopup() {
    if (!user?.emailVerified) {$('auth-dialog').showModal(); return;}
    if (paymentBusy) {$('topup-dialog').showModal(); return;}
    if (pendingPaymentId()) {await paymentReturn(); return;}
    $('topup-dialog').showModal(); paymentView('choose'); $('checkout-agree').checked = false;
    $('payment-unconfigured').hidden = !!config.payment_providers.length;
    $('payment-sandbox').hidden = config.payment_mode !== 'sandbox';
    for (const provider of ['stripe','paypal']) $(`pay-${provider}`).disabled = !config.payment_providers.includes(provider);
    try {
      const {packages} = await api('/billing/packages'); topupPackages = packages; renderPackages();
    } catch (error) {$('payment-status').textContent = t(error.message);}
  }
  async function checkout(provider) {
    if (paymentBusy || paymentState !== 'choose') return;
    if (!$('checkout-agree').checked) {$('payment-status').textContent = t('agreeRequired'); return;}
    paymentBusy = true;
    const package_id = $('topup-package').value;
    try {
      let pending = JSON.parse(sessionStorage.getItem(sessionKey()) || 'null');
      if (!pending || pending.provider !== provider || pending.package_id !== package_id || pending.payment_id) pending = {provider,package_id,idempotency_key:crypto.randomUUID()};
      sessionStorage.setItem(sessionKey(), JSON.stringify(pending));
      paymentView('verifying', 'openingCheckout');
      const session = await api('/billing/sessions', pending);
      sessionStorage.setItem(sessionKey(), JSON.stringify({...pending,...session}));
      location.assign(session.redirect_url);
    } catch (error) {paymentView('choose', error.message === 'not_found' ? 'paymentsUnavailable' : error.message);} finally {paymentBusy = false;}
  }
  function pendingPaymentId() {
    const params = new URLSearchParams(location.search);
    if (params.get('checkout') === 'cancel') return null;
    const saved = JSON.parse(sessionStorage.getItem(sessionKey()) || 'null');
    return params.get('payment_id') || saved?.payment_id;
  }
  async function paymentReturn() {
    const params = new URLSearchParams(location.search);
    if (params.get('checkout') === 'cancel') {
      sessionStorage.removeItem(sessionKey()); history.replaceState({},'',location.pathname);
      await openTopup(); paymentView('choose','checkoutCancelled'); return;
    }
    const paymentId = pendingPaymentId();
    if (!paymentId || paymentBusy) return;
    $('topup-dialog').showModal();
    paymentBusy = true;
    paymentView('verifying', 'paymentPending'); lastPaymentCheck = Date.now();
    const returnUser = user?.uid;
    try {
      const payment = await verifyReturnedPayment({paymentId, paypalToken:params.get('token'), api,
        isCurrent:() => user?.uid === returnUser});
      if (!payment || user?.uid !== returnUser) return;
      await refresh(); paymentView('succeeded', 'paymentSuccess');
      sessionStorage.removeItem(sessionKey()); history.replaceState({},'',location.pathname);
    } catch (error) {
      if (user?.uid === returnUser) paymentView('unconfirmed', ['paymentMismatch','paymentPendingLong','paymentNotCompleted'].includes(error.message) ? error.message : 'paymentCheckUnavailable');
    } finally {paymentBusy = false;}
  }
  $('topup').onclick = openTopup;
  $('pay-stripe').onclick = () => checkout('stripe'); $('pay-paypal').onclick = () => checkout('paypal');
  $('refresh-balance').onclick = () => {refresh(); paymentReturn();};
  $('payment-retry').onclick = paymentReturn;
  $('payment-done').onclick = () => $('topup-dialog').close();
  window.addEventListener('focus', () => {
    if (Date.now() - lastRefresh > 10000) refresh();
    if (user?.emailVerified && Date.now() - lastPaymentCheck > 10000) paymentReturn();
  });
  setInterval(() => {if (!document.hidden) refresh();}, 15000);
  const result = {getToken: (forceRefresh = false) => config.auth_mode === 'demo' ? Promise.resolve(config.profile === 'test' ? `fake:${user.uid}` : `local.${btoa(JSON.stringify({sub:user.uid,email:user.email,email_verified:true}))}.local`) : auth?.currentUser?.getIdToken(forceRefresh), refresh, localize, openTopup, money, get balance() {return balance;}};
  // Start after the caller has installed the token provider.
  result.start = async () => {
    if (config.auth_mode === 'demo') {
      const uid = localStorage.getItem('vt.local-user') || crypto.randomUUID(); localStorage.setItem('vt.local-user',uid);
      await update({uid,email:'Demo',emailVerified:true}); return;
    }
    if (Object.values(config.firebase || {}).some(value => !value) || !config.firebase?.apiKey) {await update(null); return;}
    try {
      const app = await import('https://www.gstatic.com/firebasejs/12.19.0/firebase-app.js');
      sdk = await import('https://www.gstatic.com/firebasejs/12.19.0/firebase-auth.js');
      auth = sdk.getAuth(app.initializeApp(config.firebase)); localize();
      let updates = Promise.resolve();
      sdk.onIdTokenChanged(auth, next => {
        updates = updates.then(() => update(next)).catch(report);
      });
    } catch {await update(null); report(new Error('authFailed'));}
  };
  localize(); return result;
}
