export function packageLabel(packageInfo, money, t) {
  const bonus = Math.round((packageInfo.point_units - packageInfo.amount_minor) * 100 / packageInfo.amount_minor);
  const label = t('topupCreditLabel').replace('{pay}', money(packageInfo.amount_minor))
    .replace('{credit}', money(packageInfo.point_units));
  return bonus > 0 ? `${label} · ${t('topupBonusLabel').replace('{percent}', bonus)}` : label;
}
