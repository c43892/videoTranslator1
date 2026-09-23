// Provider verification can be repeated safely; it never creates a new checkout.
export async function verifyReturnedPayment({paymentId, paypalToken, api, isCurrent = () => true,
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), attempts = 12, timeoutMs = 20000}) {
  const path = `/billing/payments/${encodeURIComponent(paymentId)}`;
  async function request(url, body) {
    let timer;
    try {
      return await Promise.race([api(url, body), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('paymentCheckUnavailable')), timeoutMs);
      })]);
    } finally {clearTimeout(timer);}
  }
  let payment = await request(path);
  if (!isCurrent()) return null;
  if (payment.provider === 'paypal' && payment.status === 'pending') {
    if (!paypalToken || paypalToken !== payment.provider_order_id) throw new Error('paymentMismatch');
    await request(`/billing/paypal/orders/${encodeURIComponent(payment.provider_order_id)}/capture`, {});
  }
  for (let attempt = 0; attempt < attempts && isCurrent(); attempt++) {
    if (payment.status === 'succeeded') return payment;
    if (payment.status !== 'pending') throw new Error('paymentNotCompleted');
    payment = payment.provider === 'stripe' && attempt % 4 === 0
      ? await request(path + '/reconcile', {}) : await request(path);
    if (!isCurrent()) return null;
    if (payment.status === 'succeeded') return payment;
    if (attempt < attempts - 1) await sleep(2500);
  }
  if (isCurrent()) throw new Error('paymentPendingLong');
  return null;
}
