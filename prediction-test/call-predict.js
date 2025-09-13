import { predictStatistics } from '../prediction-api-client/dist';
import listFacilities from '../data/facility-metadata.json';
var facilities = listFacilities.map(f => f.name);

facilities = ['wirkstrom'];

facilities.forEach(async (facility) => {

async function main() {
const response = await predictStatistics({body: {
  facility_name: facility,
  time_features: true,
  weather_features: true,
  rolling_features: true,
  lag_features: true,
  interaction_features: true,
  windows: [3, 7, 15, 30, 60],
  lags: [3, 7, 14, 21, 30, 45, 60],
  // custom_start_date: "2025-06-26",
  // custom_end_date: "2025-07-12",

  // XGBoost parameters
  xgb_params: {
    max_depth: 8,
    learning_rate: 0.2,
    n_estimators: 100,
    min_child_weight: 1,
    gamma: 0,
    subsample: 0.8,
    colsample_bytree: 0.8,
    reg_alpha: 0,
    reg_lambda: 1,
    random_state: 42,
    n_jobs: -1
  },

  // Prophet parameters
  prophet_params: {
    yearly_seasonality: true,
    weekly_seasonality: true,
    daily_seasonality: true,
    changepoint_prior_scale: 0.05,
    seasonality_prior_scale: 10.0,
    holidays_prior_scale: 10.0,
    seasonality_mode: 'multiplicative',
    growth: 'linear',
    n_changepoints: 25,
    changepoint_range: 0.8,
    interval_width: 0.95,
    mcmc_samples: 0
  }
}})
if (response.error) return console.log(response.error)

  console.log('Cross-validation results:\n', response.data.cross_validation_results
  .sort((a,b) => (b.accuracy || 0) - (a.accuracy || 0))
  .map(r=>r.model + " " + (r.accuracy ? r.accuracy.toFixed(1) : "N/A") + "%")
  .join('\n')
);
console.log('\nPost-training validation results:\n', response.data.post_validation_results
  .sort((a,b) => (b.accuracy || 0) - (a.accuracy || 0))
  .map(r=>r.model + " " + (r.accuracy ? r.accuracy.toFixed(1) : "N/A") + "%")
  .join('\n')
);

if (response.data.predictions_90_days && response.data.predictions_90_days.length > 0) {
  console.log('\n90-day predictions generated:', response.data.predictions_90_days.length, 'predictions');
  console.log('Date range:', response.data.predictions_90_days[0].date, 'to', response.data.predictions_90_days[response.data.predictions_90_days.length - 1].date);
  console.log('Saved to: predictions-90.csv');
}

    }

  main()
})
