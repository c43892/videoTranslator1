// Retry only explicit authentication rejections, before the endpoint can run.
// Transport failures and general 5xx responses may have committed a mutation.
export async function authenticatedFetch(url, options, {
  getToken, isCurrent, fetchImpl = fetch,
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
}) {
  let refreshed = false, transientRetries = 0, forceRefresh = false;
  for (;;) {
    if (!isCurrent()) throw new Error('unauthenticated');
    const token = await getToken(forceRefresh);
    forceRefresh = false;
    if (!token || !isCurrent()) throw new Error('unauthenticated');
    const response = await fetchImpl(url, {...options,
      headers: {...options.headers, Authorization: `Bearer ${token}`}});
    if (!isCurrent()) throw new Error('unauthenticated');
    if (![401, 503].includes(response.status)) return response;
    const data = await response.clone().json().catch(() => ({}));
    if (response.status === 401 && data.detail?.code === 'unauthenticated' && !refreshed) {
      refreshed = true; forceRefresh = true; continue;
    }
    if (response.status === 503 && data.detail?.code === 'auth_unavailable' && transientRetries < 2) {
      await sleep(500 * ++transientRetries); continue;
    }
    return response;
  }
}
