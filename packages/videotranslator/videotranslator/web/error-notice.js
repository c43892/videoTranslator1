// A recovered background request must not leave a stale sign-in warning visible.
// Successful polling must also not erase unrelated errors from user actions.
export function createErrorNotice({element, t, isRunning = () => false}) {
  let current = null, authFailures = 0;
  const authCodes = new Set(['unauthenticated', 'auth_unavailable']);
  const transientCodes = new Set(['network', 'auth_unavailable', 'unauthenticated']);
  function clear() {current = null; element.hidden = true; element.textContent = '';}
  function show(error, {background = false, source = ''} = {}) {
    const code = error.code || error.message;
    const auth = authCodes.has(code);
    if (auth) authFailures++;
    current = {code, auth, background, source};
    let message = error.message;
    if (background && isRunning() && transientCodes.has(code)) {
      message = code === 'unauthenticated' && authFailures >= 3 ? 'jobAuthRequired' : 'jobStatusReconnecting';
    }
    element.textContent = t(message); element.hidden = false;
  }
  return {
    show, clear,
    authenticated() {authFailures = 0; if (current?.auth) clear();},
    recovered(source) {
      if (current?.background && current.source === source && transientCodes.has(current.code)) clear();
    },
  };
}
