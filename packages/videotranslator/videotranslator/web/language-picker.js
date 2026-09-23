// Local SVG flags also render on Windows, where flag emoji may display as letters.
const svg = content => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">${content}</svg>`)}`;
const rect = (fill, x=0, y=0, width=30, height=20) => `<rect fill="${fill}" x="${x}" y="${y}" width="${width}" height="${height}"/>`;
const star = '<path d="M0-3 .7-.9 2.9-.9 1.1.4 1.8 2.5 0 1.2-1.8 2.5-1.1.4-2.9-.9-.7-.9Z"/>';
const flags = {
  zh: svg(rect('#de2910') + `<g fill="#ffde00"><g transform="translate(5 5)">${star}</g><g transform="translate(10 2) scale(.33)">${star}</g><g transform="translate(12 4) scale(.33)">${star}</g><g transform="translate(12 7) scale(.33)">${star}</g><g transform="translate(10 9) scale(.33)">${star}</g></g>`),
  en: svg(rect('#012169') + '<path stroke="white" stroke-width="5" d="m0 0 30 20m0-20L0 20"/><path stroke="#c8102e" stroke-width="2" d="m0 0 30 20m0-20L0 20"/><path stroke="white" stroke-width="7" d="M15 0v20M0 10h30"/><path stroke="#c8102e" stroke-width="4" d="M15 0v20M0 10h30"/>'),
};

export function createLanguagePicker(select) {
  const wrapper = document.createElement('div');
  wrapper.className = 'language-picker';
  select.before(wrapper); wrapper.append(select); select.hidden = true;
  const trigger = document.createElement('button');
  trigger.type = 'button'; trigger.className = 'language-trigger';
  trigger.setAttribute('aria-label', select.getAttribute('aria-label'));
  trigger.setAttribute('aria-haspopup', 'listbox');
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-controls', `${select.id}-options`);
  const menu = document.createElement('div');
  menu.id = `${select.id}-options`; menu.className = 'language-options'; menu.hidden = true;
  menu.setAttribute('role', 'listbox'); menu.setAttribute('aria-label', select.getAttribute('aria-label'));
  wrapper.append(trigger, menu);
  const items = [...select.options].map(option => {
    const item = document.createElement('button'); item.type = 'button'; item.tabIndex = -1;
    item.className = 'language-option'; item.setAttribute('role', 'option'); item.dataset.value = option.value;
    item.addEventListener('click', () => {
      select.value = option.value; close(); trigger.focus();
      select.dispatchEvent(new Event('change', {bubbles:true}));
    });
    menu.append(item); return item;
  });
  function close() {menu.hidden = true; trigger.setAttribute('aria-expanded','false');}
  function open(index = select.selectedIndex) {
    if (trigger.disabled) return;
    menu.hidden = false; trigger.setAttribute('aria-expanded','true');
    items[Math.max(0,index)].focus();
  }
  trigger.addEventListener('click', () => menu.hidden ? open() : close());
  trigger.addEventListener('keydown', event => {
    if (['ArrowDown','ArrowUp'].includes(event.key)) {event.preventDefault(); open();}
  });
  menu.addEventListener('keydown', event => {
    const index = items.indexOf(document.activeElement);
    if (event.key === 'Escape') {event.preventDefault(); close(); trigger.focus();}
    else if (event.key === 'Tab') close();
    else if (['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length-1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[next].focus();
    }
  });
  document.addEventListener('click', event => {if (!wrapper.contains(event.target)) close();});
  wrapper.addEventListener('focusout', event => {if (!wrapper.contains(event.relatedTarget)) close();});
  function label(element, text, code) {
    const flag = document.createElement('img'); flag.src = flags[code] || flags.en;
    flag.className = 'language-flag'; flag.alt = ''; flag.width = 21; flag.height = 14;
    const name = document.createElement('span'); name.textContent = text;
    element.replaceChildren(flag,name);
  }
  return {
    update(currentLanguage, autoLabel) {
      trigger.setAttribute('aria-label', select.getAttribute('aria-label'));
      menu.setAttribute('aria-label', select.getAttribute('aria-label'));
      select.options[0].textContent = autoLabel;
      items.forEach((item,index) => {
        const option = select.options[index];
        label(item, option.textContent, option.value === 'auto' ? currentLanguage : option.value);
        item.setAttribute('aria-selected', String(option.selected));
      });
      label(trigger, select.selectedOptions[0].textContent, select.value === 'auto' ? currentLanguage : select.value);
      const arrow = document.createElement('span'); arrow.className = 'language-chevron'; arrow.setAttribute('aria-hidden','true'); arrow.textContent = '⌄'; trigger.append(arrow);
    },
    setDisabled(disabled) {trigger.disabled = disabled; if (disabled) close();},
  };
}
