[Documentation](../modules.md) / api-client/dist

## ClientOptions

```ts
type ClientOptions = object;
```

Defined in: api-client/dist/index.d.ts:6

### Properties

#### baseUrl

```ts
baseUrl: "http://localhost:8000" | string & object;
```

Defined in: api-client/dist/index.d.ts:7

***

## FeatureConfig

```ts
type FeatureConfig = object;
```

Defined in: api-client/dist/index.d.ts:14

Feature Configuration
Configuration object for feature engineering and model training parameters

### Properties

#### time\_features?

```ts
optional time_features: boolean;
```

Defined in: api-client/dist/index.d.ts:19

Time Features
Include time-based features such as hour, day, month, seasonality

#### weather\_features?

```ts
optional weather_features: boolean;
```

Defined in: api-client/dist/index.d.ts:24

Weather Features
Include weather-related features and weather interactions

#### rolling\_features?

```ts
optional rolling_features: boolean;
```

Defined in: api-client/dist/index.d.ts:29

Rolling Features
Include rolling window statistics (mean, std, min, max)

#### lag\_features?

```ts
optional lag_features: boolean;
```

Defined in: api-client/dist/index.d.ts:34

Lag Features
Include lagged values of energy consumption

#### interaction\_features?

```ts
optional interaction_features: boolean;
```

Defined in: api-client/dist/index.d.ts:39

Interaction Features
Include feature interactions (e.g., temperature * humidity)

#### windows?

```ts
optional windows: number[];
```

Defined in: api-client/dist/index.d.ts:44

Rolling Windows
List of rolling window sizes in days for statistical features

#### lags?

```ts
optional lags: number[];
```

Defined in: api-client/dist/index.d.ts:49

Lag Periods
List of lag periods in days for historical energy values

***

## GetDocumentationData

```ts
type GetDocumentationData = object;
```

Defined in: api-client/dist/index.d.ts:58

### Properties

#### body?

```ts
optional body: never;
```

Defined in: api-client/dist/index.d.ts:59

#### path?

```ts
optional path: never;
```

Defined in: api-client/dist/index.d.ts:60

#### query?

```ts
optional query: never;
```

Defined in: api-client/dist/index.d.ts:61

#### url

```ts
url: "/docs";
```

Defined in: api-client/dist/index.d.ts:62

***

## GetDocumentationResponse

```ts
type GetDocumentationResponse = GetDocumentationResponses[keyof GetDocumentationResponses];
```

Defined in: api-client/dist/index.d.ts:65

***

## GetDocumentationResponses

```ts
type GetDocumentationResponses = object;
```

Defined in: api-client/dist/index.d.ts:67

### Properties

#### 200

```ts
200: string;
```

Defined in: api-client/dist/index.d.ts:71

API documentation page

***

## HttpError

```ts
type HttpError = object;
```

Defined in: api-client/dist/index.d.ts:78

HTTP Error
HTTP error response

### Properties

#### detail

```ts
detail: string;
```

Defined in: api-client/dist/index.d.ts:83

Error Detail
Detailed error message

***

## ModelMetrics

```ts
type ModelMetrics = object;
```

Defined in: api-client/dist/index.d.ts:90

Model Performance Metrics
Performance metrics for a trained model

### Properties

#### model

```ts
model: "Prophet" | "Random Forest" | "Ridge Regression" | "Ensemble";
```

Defined in: api-client/dist/index.d.ts:95

Model Name
Name of the machine learning model

#### mae

```ts
mae: number;
```

Defined in: api-client/dist/index.d.ts:100

Mean Absolute Error
Mean Absolute Error in millions of energy units

#### rmse

```ts
rmse: number;
```

Defined in: api-client/dist/index.d.ts:105

Root Mean Square Error
Root Mean Square Error in millions of energy units

#### r2\_score?

```ts
optional r2_score: number | null;
```

Defined in: api-client/dist/index.d.ts:110

R-squared Score
Coefficient of determination (R²) indicating model fit quality

#### accuracy?

```ts
optional accuracy: number | null;
```

Defined in: api-client/dist/index.d.ts:115

Accuracy Percentage
Model accuracy as percentage (100 - MAPE)

***

## Options&lt;TData, ThrowOnError&gt;

```ts
type Options<TData, ThrowOnError> = Options_2<TData, ThrowOnError> & object;
```

Defined in: api-client/dist/index.d.ts:118

### Type declaration

<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Defined in</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`client?`

</td>
<td>

`Client`

</td>
<td>

You can provide a client instance returned by `createClient()` instead of
individual options. This might be also useful if you want to implement a
custom client.

</td>
<td>

api-client/dist/index.d.ts:124

</td>
</tr>
<tr>
<td>

`meta?`

</td>
<td>

`Record`&lt;`string`, `unknown`&gt;

</td>
<td>

You can pass arbitrary values through the `meta` object. This can be
used to access values that aren't defined as part of the SDK function.

</td>
<td>

api-client/dist/index.d.ts:129

</td>
</tr>
</tbody>
</table>

### Type Parameters

<table>
<thead>
<tr>
<th>Type Parameter</th>
<th>Default type</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`TData` *extends* `TDataShape`

</td>
<td>

`TDataShape`

</td>
</tr>
<tr>
<td>

`ThrowOnError` *extends* `boolean`

</td>
<td>

`boolean`

</td>
</tr>
</tbody>
</table>

***

## PredictionResponse

```ts
type PredictionResponse = object;
```

Defined in: api-client/dist/index.d.ts:136

Prediction Response
Complete response containing predictions and model performance metrics

### Properties

#### predictions

```ts
predictions: PredictionResult[];
```

Defined in: api-client/dist/index.d.ts:141

Predictions
List of energy predictions for each date

#### cross\_validation\_results

```ts
cross_validation_results: ModelMetrics[];
```

Defined in: api-client/dist/index.d.ts:146

Cross-Validation Results
Performance metrics from cross-validation testing

#### may\_validation\_results

```ts
may_validation_results: ModelMetrics[];
```

Defined in: api-client/dist/index.d.ts:151

May Validation Results
Performance metrics from May validation data

#### feature\_config

```ts
feature_config: FeatureConfig;
```

Defined in: api-client/dist/index.d.ts:152

***

## PredictionResult

```ts
type PredictionResult = object;
```

Defined in: api-client/dist/index.d.ts:159

Prediction Result
Energy prediction result for a single date

### Properties

#### date

```ts
date: string;
```

Defined in: api-client/dist/index.d.ts:164

Date
Prediction date in YYYY-MM-DD format

#### predicted\_energy\_millions

```ts
predicted_energy_millions: number;
```

Defined in: api-client/dist/index.d.ts:169

Final Prediction
Final ensemble prediction in millions of energy units

#### prophet\_prediction

```ts
prophet_prediction: number;
```

Defined in: api-client/dist/index.d.ts:174

Prophet Prediction
Prophet model prediction in millions of energy units

#### rf\_prediction

```ts
rf_prediction: number;
```

Defined in: api-client/dist/index.d.ts:179

Random Forest Prediction
Random Forest model prediction in millions of energy units

#### ridge\_prediction

```ts
ridge_prediction: number;
```

Defined in: api-client/dist/index.d.ts:184

Ridge Regression Prediction
Ridge Regression model prediction in millions of energy units

#### ensemble\_prediction

```ts
ensemble_prediction: number;
```

Defined in: api-client/dist/index.d.ts:189

Ensemble Prediction
Average of all model predictions in millions of energy units

#### prediction\_lower

```ts
prediction_lower: number;
```

Defined in: api-client/dist/index.d.ts:194

Lower Confidence Bound
Lower bound of prediction confidence interval

#### prediction\_upper

```ts
prediction_upper: number;
```

Defined in: api-client/dist/index.d.ts:199

Upper Confidence Bound
Upper bound of prediction confidence interval

#### actual\_energy\_millions

```ts
actual_energy_millions: number;
```

Defined in: api-client/dist/index.d.ts:204

Actual Energy
Actual energy consumption in millions of energy units

#### error

```ts
error: number;
```

Defined in: api-client/dist/index.d.ts:209

Prediction Error
Difference between actual and predicted values (actual - predicted)

#### percent\_error

```ts
percent_error: number;
```

Defined in: api-client/dist/index.d.ts:214

Percentage Error
Absolute percentage error

#### weather\_data

```ts
weather_data: WeatherData;
```

Defined in: api-client/dist/index.d.ts:215

***

## PredictStatisticsData

```ts
type PredictStatisticsData = object;
```

Defined in: api-client/dist/index.d.ts:224

### Properties

#### body

```ts
body: FeatureConfig;
```

Defined in: api-client/dist/index.d.ts:225

#### path?

```ts
optional path: never;
```

Defined in: api-client/dist/index.d.ts:226

#### query?

```ts
optional query: never;
```

Defined in: api-client/dist/index.d.ts:227

#### url

```ts
url: "/predict";
```

Defined in: api-client/dist/index.d.ts:228

***

## PredictStatisticsError

```ts
type PredictStatisticsError = PredictStatisticsErrors[keyof PredictStatisticsErrors];
```

Defined in: api-client/dist/index.d.ts:231

***

## PredictStatisticsErrors

```ts
type PredictStatisticsErrors = object;
```

Defined in: api-client/dist/index.d.ts:233

### Properties

#### 422

```ts
422: ValidationError;
```

Defined in: api-client/dist/index.d.ts:237

Validation Error

#### 500

```ts
500: HttpError;
```

Defined in: api-client/dist/index.d.ts:241

Internal Server Error

***

## PredictStatisticsResponse

```ts
type PredictStatisticsResponse = PredictStatisticsResponses[keyof PredictStatisticsResponses];
```

Defined in: api-client/dist/index.d.ts:244

***

## PredictStatisticsResponses

```ts
type PredictStatisticsResponses = object;
```

Defined in: api-client/dist/index.d.ts:246

### Properties

#### 200

```ts
200: PredictionResponse;
```

Defined in: api-client/dist/index.d.ts:250

Successful prediction generation

***

## ValidationError

```ts
type ValidationError = object;
```

Defined in: api-client/dist/index.d.ts:257

Validation Error
Request validation error details

### Properties

#### detail

```ts
detail: object[];
```

Defined in: api-client/dist/index.d.ts:261

Error Details

<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Defined in</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`loc`

</td>
<td>

(`string` \| `number`)[]

</td>
<td>

Location
Location of the validation error

</td>
<td>

api-client/dist/index.d.ts:266

</td>
</tr>
<tr>
<td>

`msg`

</td>
<td>

`string`

</td>
<td>

Message
Error message

</td>
<td>

api-client/dist/index.d.ts:271

</td>
</tr>
<tr>
<td>

`type`

</td>
<td>

`string`

</td>
<td>

Error Type
Type of validation error

</td>
<td>

api-client/dist/index.d.ts:276

</td>
</tr>
</tbody>
</table>

***

## WeatherData

```ts
type WeatherData = object;
```

Defined in: api-client/dist/index.d.ts:284

Weather Data
Weather information for a specific date

### Properties

#### temperature\_2m\_mean?

```ts
optional temperature_2m_mean: number | null;
```

Defined in: api-client/dist/index.d.ts:289

Average Temperature
Mean temperature at 2 meters height in Celsius

#### soil\_moisture\_0\_to\_7cm\_mean?

```ts
optional soil_moisture_0_to_7cm_mean: number | null;
```

Defined in: api-client/dist/index.d.ts:294

Soil Moisture
Average soil moisture in top 7cm layer

#### precipitation\_sum?

```ts
optional precipitation_sum: number | null;
```

Defined in: api-client/dist/index.d.ts:299

Precipitation
Total precipitation in mm

#### relative\_humidity\_2m\_mean?

```ts
optional relative_humidity_2m_mean: number | null;
```

Defined in: api-client/dist/index.d.ts:304

Relative Humidity
Average relative humidity at 2 meters height as percentage

***

## getDocumentation()

```ts
const getDocumentation: <ThrowOnError>(options?: Options<GetDocumentationData, ThrowOnError>) => RequestResult<GetDocumentationResponses, unknown, ThrowOnError, "fields">;
```

Defined in: api-client/dist/index.d.ts:56

API Documentation
Interactive API documentation (Swagger UI)

### Type Parameters

<table>
<thead>
<tr>
<th>Type Parameter</th>
<th>Default type</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`ThrowOnError` *extends* `boolean`

</td>
<td>

`false`

</td>
</tr>
</tbody>
</table>

### Parameters

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`options?`

</td>
<td>

[`Options`](#options)&lt;[`GetDocumentationData`](#getdocumentationdata), `ThrowOnError`&gt;

</td>
</tr>
</tbody>
</table>

### Returns

`RequestResult`&lt;[`GetDocumentationResponses`](#getdocumentationresponses), `unknown`, `ThrowOnError`, `"fields"`&gt;

***

## predictStatistics()

```ts
const predictStatistics: <ThrowOnError>(options: Options<PredictStatisticsData, ThrowOnError>) => RequestResult<PredictStatisticsResponses, PredictStatisticsErrors, ThrowOnError, "fields">;
```

Defined in: api-client/dist/index.d.ts:222

Generate Statistic Predictions
Trains multiple machine learning models (Prophet, Random Forest, Ridge Regression) and generates energy consumption predictions with comprehensive validation metrics. The API performs feature engineering, model training, cross-validation, and final predictions for specified dates.

### Type Parameters

<table>
<thead>
<tr>
<th>Type Parameter</th>
<th>Default type</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`ThrowOnError` *extends* `boolean`

</td>
<td>

`false`

</td>
</tr>
</tbody>
</table>

### Parameters

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td>

`options`

</td>
<td>

[`Options`](#options)&lt;[`PredictStatisticsData`](#predictstatisticsdata), `ThrowOnError`&gt;

</td>
</tr>
</tbody>
</table>

### Returns

`RequestResult`&lt;[`PredictStatisticsResponses`](#predictstatisticsresponses), [`PredictStatisticsErrors`](#predictstatisticserrors), `ThrowOnError`, `"fields"`&gt;
