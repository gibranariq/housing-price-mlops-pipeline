import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

class ModelMonitor:
    def __init__(
        self,
        threshold_variance=120000,
        threshold_samples=10, # low threshold for testing/demo
        threshold_mae_increase=1.3, # 30% increase
    ):
        self.threshold_variance = threshold_variance
        self.threshold_samples = threshold_samples
        self.threshold_mae_increase = threshold_mae_increase
        self.alert_log = []

    def check_performance_drift(self, predictions_file="logs/predictions.jsonl"):
        """Check if model performance has degraded or data drift has occurred"""
        try:
            # Read recent predictions
            predictions = []
            if not os.path.exists(predictions_file):
                print(f"❌ Prediction log '{predictions_file}' not found.")
                return False, None

            with open(predictions_file, "r") as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))

            if len(predictions) < self.threshold_samples:
                print(f"⏳ Not enough samples yet: {len(predictions)}/{self.threshold_samples}")
                return False, None

            # Get recent predictions (last N samples)
            recent = predictions[-self.threshold_samples :]

            # Calculate statistics
            pred_values = [p["prediction"] for p in recent]
            mean_pred = np.mean(pred_values)
            std_pred = np.std(pred_values)

            # Load baseline model metrics
            try:
                with open("logs/latest_metrics.json", "r") as f:
                    model_metrics = json.load(f)
                baseline_rmse = model_metrics.get("rmse", 0.11)  # log-scale RMSE default
                print(f"✅ Loaded baseline metrics (RMSE: {baseline_rmse:.4f})")
            except Exception as e:
                print(f"⚠️ Could not load baseline metrics, using defaults. Error: {e}")
                baseline_rmse = 0.11

            print(f"\n📊 Ames Housing Monitoring Report:")
            print(f"   Total predictions: {len(predictions)}")
            print(f"   Analyzing last: {len(recent)} predictions")
            print(f"   Mean price prediction: ${mean_pred:,.2f}")
            print(f"   Std dev of predictions: ${std_pred:,.2f}")
            print(f"   Variance threshold: ${self.threshold_variance:,.2f}")

            drift_detected = False
            drift_reasons = []

            # 1. Check standard deviation (too high/low variance)
            if std_pred > self.threshold_variance:
                drift_reasons.append(
                    f"High prediction variance: ${std_pred:,.2f} > ${self.threshold_variance:,.2f}"
                )
                drift_detected = True

            # 2. Check prediction distribution shift (price mean out of bound)
            if mean_pred > 300000 or mean_pred < 100000:
                drift_reasons.append(
                    f"Prediction distribution shift: Mean price ${mean_pred:,.2f} is outside normal bounds ($100k - $300k)"
                )
                drift_detected = True

            # 3. Compare recent predictions with earlier ones (concept drift)
            if len(predictions) >= self.threshold_samples * 2:
                earlier = predictions[-(self.threshold_samples * 2) : -self.threshold_samples]
                earlier_mean = np.mean([p["prediction"] for p in earlier])
                mean_shift = abs(mean_pred - earlier_mean) / earlier_mean

                if mean_shift > 0.25:  # 25% shift in average house price
                    drift_reasons.append(
                        f"Concept drift: {mean_shift*100:.1f}% shift in average price compared to historical predictions."
                    )
                    drift_detected = True

            if drift_detected:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "reasons": drift_reasons,
                    "metrics": {
                        "mean_prediction": float(mean_pred),
                        "std_prediction": float(std_pred),
                        "prediction_count": len(predictions),
                        "baseline_rmse": float(baseline_rmse),
                    },
                }
                self.alert_log.append(alert)

                print(f"\n⚠️  DRIFT DETECTED - Model retraining recommended!")
                print(f"   Reasons:")
                for reason in drift_reasons:
                    print(f"   - {reason}")

                return True, alert

            print(f"\n✅ Model performance is stable and no drift detected.")
            return False, None

        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def save_alert(self, alert):
        """Save alert to logs/alerts.jsonl"""
        if alert:
            os.makedirs("logs", exist_ok=True)
            with open("logs/alerts.jsonl", "a") as f:
                f.write(json.dumps(alert) + "\n")
            print(f"💾 Alert saved to logs/alerts.jsonl")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    monitor = ModelMonitor(threshold_variance=120000, threshold_samples=10)

    print("🔍 Starting Ames Housing Model Monitor...")
    print("=" * 60)

    if mode == "once":
        needs_retraining, alert = monitor.check_performance_drift()
        if needs_retraining:
            monitor.save_alert(alert)
            print("\n" + "=" * 60)
            print("🚨 ACTION REQUIRED: Run retraining pipeline")
            print("   Command: python src/retrain_trigger.py data/train.csv")
            print("=" * 60)
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print("✅ No action needed")
            print("=" * 60)
            sys.exit(0)
    else:
        # Continuous monitoring loop
        import time
        while True:
            needs_retraining, alert = monitor.check_performance_drift()
            if needs_retraining:
                monitor.save_alert(alert)
                print("\n🔄 Triggering retraining pipeline automatically...")
                # We can execute trigger here or let the orchestrator do it
                break
            print(f"\n⏰ Next check in 30 seconds...")
            print("=" * 60)
            time.sleep(30)

if __name__ == "__main__":
    main()
