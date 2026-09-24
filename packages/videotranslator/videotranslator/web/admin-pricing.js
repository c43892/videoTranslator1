// Access is checked again by the server using the verified Firebase UID.
export function pricingText(key, price, language) {
  if (!price) return key === 'rateNotice' ? '…' : '';
  const money = n => `US$${(n / 100).toFixed(2)}`;
  const rate = money(price.rate_cents_per_minute);
  if (key === 'rateNotice') return language === 'zh' ? `视频与音频 · 每分钟 ${rate}` : `Video and audio · ${rate} / minute`;
  const minimum = money(price.minimum_cents), increment = money(price.billing_increment_cents);
  return language === 'zh' ? `按实际时长以每分钟 ${rate} 计费，总额向上取整至 ${increment} 的整数倍，最低 ${minimum}。音频与视频同价。` :
    `Charged by actual duration at ${rate} per minute, total rounded up in ${increment} increments (minimum ${minimum}). Audio and video cost the same.`;
}

export function createAdminPricing({api, language, updated}) {
  const zh = () => language() === 'zh';
  const button = document.createElement('button'); button.type = 'button'; button.hidden = true;
  button.id = 'admin-pricing';
  document.getElementById('signout').before(button);
  const dialog = document.createElement('dialog'); dialog.className = 'account-dialog';
  dialog.setAttribute('aria-labelledby', 'admin-pricing-title');
  dialog.innerHTML = `<form><h2 id="admin-pricing-title"></h2><p data-help></p>
    <label><span data-label="rate"></span><input name="rate" type="number" min="0.01" max="100" step="0.01" required></label>
    <label><span data-label="minimum"></span><input name="minimum" type="number" min="0.01" max="100" step="0.01" required></label>
    <label><span data-label="increment"></span><input name="increment" type="number" min="0.01" max="100" step="0.01" required></label>
    <p role="status" aria-live="polite"></p><button type="submit"></button><button type="button" data-close></button></form>`;
  document.body.append(dialog);
  let version, generation = 0;
  const form = dialog.querySelector('form'), status = dialog.querySelector('[role=status]');
  function localize() {
    button.textContent = zh() ? '价格管理' : 'Manage pricing';
    dialog.querySelector('h2').textContent = button.textContent;
    dialog.querySelector('[data-help]').textContent = zh() ? '单位：美元。保存后新报价立即生效，已有报价与任务保持原价。' : 'Amounts in USD. Saving applies immediately to new quotes. Existing quotes and jobs keep their price.';
    for (const [key, en, cn] of [['rate','Price per minute','每分钟价格'],['minimum','Minimum charge','最低收费'],['increment','Rounding increment','向上取整单位']]) dialog.querySelector(`[data-label=${key}]`).textContent = zh() ? cn : en;
    form.querySelector('[type=submit]').textContent = zh() ? '保存并生效' : 'Save and apply';
    dialog.querySelector('[data-close]').textContent = zh() ? '关闭' : 'Close';
  }
  function fill(price) {
    version = price.pricing_version;
    for (const [name,key] of [['rate','rate_cents_per_minute'],['minimum','minimum_cents'],['increment','billing_increment_cents']]) form.elements[name].value = (price[key] / 100).toFixed(2);
  }
  document.addEventListener('account-updated', e => {
    if (button.hidden !== !e.detail?.isAdmin) generation++;
    button.hidden = !e.detail?.isAdmin;
    if (button.hidden) {dialog.close(); version = undefined;}
  });
  dialog.querySelector('[data-close]').onclick = () => dialog.close();
  button.onclick = async () => {
    const current = generation; button.disabled = true;
    try {const price = await api('/admin/pricing'); if (current !== generation) return; fill(price); localize(); status.textContent = ''; dialog.showModal();}
    catch {if (current === generation) {status.textContent = zh() ? '无法加载价格，请关闭后重试。' : 'Unable to load prices. Close and try again.'; dialog.showModal();}}
    finally {button.disabled = false;}
  };
  form.onsubmit = async event => {
    event.preventDefault(); if (!version) return;
    const current = generation, submit = form.querySelector('[type=submit]'); submit.disabled = true;
    const cents = name => Math.round(Number(form.elements[name].value) * 100);
    try {
      const price = await api('/admin/pricing', {expected_version:version, rate_cents_per_minute:cents('rate'), minimum_cents:cents('minimum'), billing_increment_cents:cents('increment')}, 'PUT');
      if (current !== generation) return;
      fill(price); updated(price); status.textContent = zh() ? '已保存，新报价立即生效。' : 'Saved. New quotes now use this price.';
    } catch (error) {
      if (current !== generation) return;
      status.textContent = error.code === 'stale_job_version' ? (zh() ? '价格已被更新，请关闭后重新打开再修改。' : 'Prices changed. Close and reopen before editing.') : (zh() ? '保存失败，请重试。' : 'Save failed. Please try again.');
    } finally {submit.disabled = false;}
  };
  localize(); return {localize};
}
