/** Mirrors the backend's two-proportion z-test (app/services/stats_engine.py) so the
 * landing page demo shows genuinely computed statistics, not fake numbers. */

function normalCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  if (z > 0) prob = 1 - prob;
  return prob;
}

export function twoProportionZTest(controlSuccess: number, controlTotal: number, treatmentSuccess: number, treatmentTotal: number) {
  if (controlTotal === 0 || treatmentTotal === 0) {
    return { pValue: 1, uplift: 0, pControl: 0, pTreatment: 0 };
  }
  const pControl = controlSuccess / controlTotal;
  const pTreatment = treatmentSuccess / treatmentTotal;
  const pPool = (controlSuccess + treatmentSuccess) / (controlTotal + treatmentTotal);
  const se = Math.sqrt(pPool * (1 - pPool) * (1 / controlTotal + 1 / treatmentTotal));

  if (se === 0) {
    return { pValue: 1, uplift: 0, pControl, pTreatment };
  }

  const z = (pTreatment - pControl) / se;
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));
  const uplift = pControl > 0 ? ((pTreatment - pControl) / pControl) * 100 : 0;

  return { pValue, uplift, pControl, pTreatment };
}
