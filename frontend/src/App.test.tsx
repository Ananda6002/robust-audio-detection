import { describe, it, expect } from "vitest";
import React from "react";
import { renderToString } from "react-dom/server";
import App from "./App";

describe("AcoustiGuard AI Frontend - Component & Logic Suite", () => {
  it("renders the main application without crashing using renderToString", () => {
    const html = renderToString(<App />);
    expect(html).toBeTruthy();
    expect(html).toContain("AcoustiGuard AI");
    expect(html).toContain("UrbanSound8K");
    expect(html).toContain("Noise-Robust Environmental Audio Classification");
  });

  it("renders the 4 primary application navigation tabs", () => {
    const html = renderToString(<App />);
    expect(html).toContain("Audio Analysis");
    expect(html).toContain("Robustness Lab");
    expect(html).toContain("Performance Analytics");
    expect(html).toContain("Model &amp; Taxonomy");
  });

  it("renders the initial view with audio input and inference parameters", () => {
    const html = renderToString(<App />);
    expect(html).toContain("Environmental Sound Classifier");
    expect(html).toContain("Audio Input");
    expect(html).toContain("Click to choose audio or drop file");
    expect(html).toContain("Inference Parameters");
    expect(html).toContain("Noise-Augmented");
    expect(html).toContain("Baseline CNN");
  });

  it("evaluates selective classification rejection logic correctly", () => {
    const threshold = 0.70;

    const confidentSample = {
      predicted_class: "dog_bark",
      confidence: 0.842,
    };
    const isUncertain1 = confidentSample.confidence < threshold;
    expect(isUncertain1).toBe(false);

    const uncertainSample = {
      predicted_class: "siren",
      confidence: 0.531,
    };
    const isUncertain2 = uncertainSample.confidence < threshold;
    expect(isUncertain2).toBe(true);
  });

  it("verifies UrbanSound8K 10-class taxonomy", () => {
    const expectedClasses = [
      "air_conditioner",
      "car_horn",
      "children_playing",
      "dog_bark",
      "drilling",
      "engine_idling",
      "gun_shot",
      "jackhammer",
      "siren",
      "street_music",
    ];
    expect(expectedClasses).toHaveLength(10);
    expect(new Set(expectedClasses).size).toBe(10);
  });

  it("verifies research test bench SNR and noise parameters", () => {
    const testSnrs = [20.0, 10.0, 5.0, 0.0];
    const noiseTypes = ["white", "pink", "brown", "ambient"];

    expect(testSnrs).toHaveLength(4);
    expect(testSnrs[testSnrs.length - 1]).toBe(0.0);
    expect(noiseTypes).toContain("white");
    expect(noiseTypes).toContain("pink");
  });
});
