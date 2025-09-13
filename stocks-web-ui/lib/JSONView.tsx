"use client";
import JsonView from "@uiw/react-json-view";

export default function JSONView({ data }: { data: object }) {
  return <JsonView value={data} />;
}
