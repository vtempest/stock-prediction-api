"use client"

import type React from "react"

import { loadingPacman } from "grab-api.js/icons"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, CheckCircle, AlertCircle } from "lucide-react"

interface ModelTrainingFormProps {
  onSubmit: (data: any) => void
  status: "idle" | "training" | "success" | "error"
  result: any
}

export default function ModelTrainingForm({ onSubmit, status, result }: ModelTrainingFormProps) {
  const [formData, setFormData] = useState({
    folder: "../data/facility_1",
    datapath: "facility_1_data.csv",
    targetName: "energy_output",
    nrounds: 10000,
    statisticalAnalysis: "mean", // mean, std, rolling_7d, rolling_30d
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

    const trainingData = {
      folder: formData.folder,
      datapath: formData.datapath,
      featuresToUse: formData.featuresToUse,
      targetName: formData.targetName,
      statisticalAnalysis: formData.statisticalAnalysis,
      xgbParams: {
        nrounds: formData.nrounds,
      },
    }

    onSubmit(trainingData)
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
          <Label htmlFor="datapath">Data File Path</Label>
          <Input
            id="datapath"
            value={formData.datapath}
            onChange={(e) => setFormData((prev) => ({ ...prev, datapath: e.target.value }))}
            placeholder="facility_1_data.csv"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="targetName">Target Variable</Label>
          <Input
            id="targetName"
            value={formData.targetName}
            onChange={(e) => setFormData((prev) => ({ ...prev, targetName: e.target.value }))}
            placeholder="energy_output"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="nrounds">Training Rounds</Label>
          <Input
            id="nrounds"
            type="number"
            value={formData.nrounds}
            onChange={(e) => setFormData((prev) => ({ ...prev, nrounds: Number.parseInt(e.target.value) }))}
            placeholder="10000"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label>Statistical Analysis Method</Label>
        <Select
          value={formData.statisticalAnalysis}
          onValueChange={(value) => setFormData((prev) => ({ ...prev, statisticalAnalysis: value }))}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select analysis method" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="mean">Mean</SelectItem>
            <SelectItem value="std">Standard Deviation</SelectItem>
            <SelectItem value="rolling_7d">Rolling Average (7 days)</SelectItem>
            <SelectItem value="rolling_30d">Rolling Average (30 days)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-3">
        <Label>Weather Features to Use</Label>
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
          <AlertDescription>Training failed. Please check your data paths and try again.</AlertDescription>
        </Alert>
      )}

      {status === "success" && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>Model trained successfully! You can now use it for predictions.</AlertDescription>
        </Alert>
      )}

      <Button type="submit" disabled={result?.isLoading || formData.featuresToUse.length === 0} className="w-full">
        {result?.isLoading ? (
            <>
             {loadingPacman({
              colors: ["#FF0000", "#00FF00", "#0000FF",
                  "#FFA500", "#800080", "#FFFF00", "#00FFFF",
                  "#FF00FF", "#008000", "#FFC0CB", "#4B0082",
                  "#FF4500", "#9400D3", "#00FF7F", "#FF1493"],
              size: Math.random() * 10,
              raw: true,
            })} 
              Training Model...
          </>
        ) : (
          "Train Model"
        )}
      </Button>
    </form>
  )
}
