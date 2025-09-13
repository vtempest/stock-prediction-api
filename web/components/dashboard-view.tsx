"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import grab from "grab-api.js"
import StockPriceComparisonChart from "@/components/stock-price-comparison-chart"
import StockStatusCards from "@/components/stock-status-cards"
import ModelTrainingForm from "@/components/model-training-form"
import PredictionForm from "@/components/prediction-form"

export default function DashboardView() {
  const [activeTab, setActiveTab] = useState("dashboard")
  const [predictionStatus, setPredictionStatus] = useState<"idle" | "predicting" | "success" | "error">("idle")
  const [predictionResult, setPredictionResult] = useState<{
    predictions: {
      date: string,
      predicted_price: number,
      actual_price: number,
      error: number,
      percent_error: number,
    }[] | null;
    isLoading: boolean,
  }>({
    isLoading: false,
    predictions: null,
  })

  const [trainingResult, setTrainingResult] = useState<{
    accuracy?: number,
    trainingTime?: number,
    modelPath?: string,
    isLoading?: boolean,
  }>({})

  // @ts-ignore
  grab.defaults.debug = true;
  grab.mock["/api/train"] = {
    response: {
      success: true,
      message: "Model trained successfully",
      modelPath: `/stock_model.xgb`,
      accuracy: 0.55 + Math.random() * 0.1, // Random accuracy between 0.85-0.95
      trainingTime: Math.floor(Math.random() * 30) + 10, // Random time between 10-40s
    },
    delay: 1000,
  }

  const handleTraining = async (trainData: object) => {
    let res = await grab("train", {
      cache: false,
      response: setTrainingResult,
      ...trainData
    })
    setTrainingResult(res as any)
  }

  const handlePrediction = async (predictionData: any) => {
    const result = await grab("predict", {
      cache: false,
      body: predictionData,
      method: "POST",
      response: setPredictionResult
    })
    setPredictionResult(result as any)
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 bg-blue-700 text-white shadow-md">
        <div className="container mx-auto p-4">
          <h1 className="text-2xl font-bold">Stock Prediction Dashboard</h1>
        </div>
      </header>

      <div className="container mx-auto p-4 flex-1">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="train">Train Model</TabsTrigger>
            <TabsTrigger value="predict">Predict</TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Stock Price Comparison</CardTitle>
                  <CardDescription>Historical stock price comparison for selected stocks</CardDescription>
                </CardHeader>
                <CardContent className="h-80">
                  <StockPriceComparisonChart />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Stock Status</CardTitle>
                  <CardDescription>Current stock prices and key indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <StockStatusCards />
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>Latest model training and prediction results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-medium">Last Training Session</h4>
                    {trainingResult ? (
                      <div className="bg-blue-50 p-3 rounded border">
                        <p className="text-sm text-blue-800">
                          Model trained successfully with {(trainingResult?.accuracy ?? 0 * 100).toFixed(1)}% accuracy
                        </p>
                        <p className="text-xs text-blue-600">Training time: {trainingResult?.trainingTime}s</p>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No training sessions yet</p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-medium">Last Prediction</h4>
                    {predictionResult ? (
                      <div className="bg-green-50 p-3 rounded border">
                        <p className="text-sm text-green-800">
                          Generated {predictionResult.predictions?.length || 0} predictions
                        </p>
                        <p className="text-xs text-green-600">
                          Latest prediction: {predictionResult.predictions?.[0]?.predicted_price?.toFixed(2)} USD
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No predictions yet</p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="train" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Train Prediction Model</CardTitle>
                <CardDescription>
                  Train an XGBoost model using historical stock price and technical indicator data
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ModelTrainingForm onSubmit={handleTraining} status={"idle"} result={trainingResult} />
              </CardContent>
            </Card>

            {trainingResult.accuracy !== undefined && (
              <Card>
                <CardHeader>
                  <CardTitle>Training Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-blue-50 p-4 rounded border">
                      <h4 className="font-medium text-blue-800">Accuracy</h4>
                      <p className="text-2xl font-bold text-blue-900">{(trainingResult?.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-green-50 p-4 rounded border">
                      <h4 className="font-medium text-green-800">Training Time</h4>
                      <p className="text-2xl font-bold text-green-900">{trainingResult.trainingTime}s</p>
                    </div>
                    <div className="bg-purple-50 p-4 rounded border">
                      <h4 className="font-medium text-purple-800">Model Path</h4>
                      <p className="text-sm text-purple-900 break-all">{trainingResult.modelPath}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="predict" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Generate Predictions</CardTitle>
                <CardDescription>Use trained models to predict future stock prices</CardDescription>
              </CardHeader>
              <CardContent>
                <PredictionForm onSubmit={handlePrediction} status={predictionStatus} result={predictionResult} />
              </CardContent>
            </Card>

            {predictionResult?.isLoading && (
              <Card>
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="bg-green-50 p-4 rounded border">
                      <h4 className="font-medium text-green-800">
                        Generated {predictionResult.predictions?.length || 0} predictions
                      </h4>
                    </div>

                    <div className="max-h-64 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-muted">
                          <tr>
                            <th className="text-left p-2">Date</th>
                            <th className="text-right p-2">Predicted Price (USD)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {predictionResult.predictions?.map((pred: any, index: number) => (
                            <tr key={index} className={index % 2 === 0 ? "bg-white" : "bg-muted/50"}>
                              <td className="p-2">{pred.date}</td>
                              <td className="text-right p-2">{pred.predicted_price?.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
