"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, CheckCircle, AlertCircle } from "lucide-react"

interface PredictionFormProps {
  onSubmit: (data: any) => void
  status: "idle" | "predicting" | "success" | "error"
  result: any
}

export default function PredictionForm({ onSubmit, status, result }: PredictionFormProps) {
  const [formData, setFormData] = useState({
    folder: "../data/facility_1",
    modelPath: "../data/facility_1/xgboost_model.xgb",
    featuresToUse: [
      "temperature_2m_mean",
      "precipitation_sum",
      "soil_temperature_0_to_7cm_mean",
      "soil_moisture_0_to_7cm_mean",
      "relative_humidity_2m_mean",
    ],
  })

  const availableFeatures = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "soil_temperature_0_to_7cm_mean",
    "soil_moisture_0_to_7cm_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
  ]

  const handleFeatureToggle = (feature: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      featuresToUse: checked ? [...prev.featuresToUse, feature] : prev.featuresToUse.filter((f) => f !== feature),
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="folder">Data Folder Path</Label>
          <Input
            id="folder"
            value={formData.folder}
            onChange={(e) => setFormData((prev) => ({ ...prev, folder: e.target.value }))}
            placeholder="../data/facility_1"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="modelPath">Model Path</Label>
          <Input
            id="modelPath"
            value={formData.modelPath}
            onChange={(e) => setFormData((prev) => ({ ...prev, modelPath: e.target.value }))}
            placeholder="../data/facility_1/xgboost_model.xgb"
          />
        </div>
      </div>

      <div className="space-y-3">
        <Label>Weather Features for Prediction</Label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {availableFeatures.map((feature) => (
            <div key={feature} className="flex items-center space-x-2">
              <Checkbox
                id={feature}
                checked={formData.featuresToUse.includes(feature)}
                onCheckedChange={(checked) => handleFeatureToggle(feature, checked as boolean)}
              />
              <Label htmlFor={feature} className="text-sm">
                {feature.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {status === "error" && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Prediction failed. Please check your model path and try again.</AlertDescription>
        </Alert>
      )}

      {status === "success" && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>Predictions generated successfully!</AlertDescription>
        </Alert>
      )}

      <Button
        type="submit"
        disabled={status === "predicting" || formData.featuresToUse.length === 0}
        className="w-full"
      >
        {status === "predicting" ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Generating Predictions...
          </>
        ) : (
          "Generate Predictions"
        )}
      </Button>
    </form>
  )
}
