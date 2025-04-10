import json
import os


def load_json_file(file_path):
    """
    Lädt eine JSON-Datei und gibt die Daten zurück.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_all_json_files_rag(folder):
    return [
        (filename, json.load(open(os.path.join(folder, filename), encoding="utf-8")))
        for filename in os.listdir(folder)
        if filename.lower().startswith("testdatensatz-rag-synth")
    ]

def load_all_json_files(folder):
    return [
        (filename, json.load(open(os.path.join(folder, filename), encoding="utf-8")))
        for filename in os.listdir(folder)
        if filename.lower().endswith(".json")
    ]


def get_predictions_path(base_folder, llm_name, filename):
    """
    Gibt den vollständigen Pfad zu einer Predictidateion zurück.
    """
    return os.path.normpath(
        os.path.join(base_folder, llm_name, "predictions", filename)
    )


def calculate_metrics(ground_truth, predictions):
    """
    Berechnet die Metriken (TP, FP, FN) für die Prediction und gibt die Ergebnisse zurück.
    """
    results = {
        "from": {"TP": 0, "FP": 0, "FN": 0, "errors": []},
        "to": {"TP": 0, "FP": 0, "FN": 0, "errors": []},
        "date": {"TP": 0, "FP": 0, "FN": 0, "errors": []},
        "time": {"TP": 0, "FP": 0, "FN": 0, "errors": []},
    }

    for gt, pred in zip(ground_truth, predictions):
        for key in results.keys():
            gt_value = gt["entitys"].get(key, None)
            pred_value = pred["entitys"].get(key, None)

            if gt_value == pred_value and gt_value is not None:
                results[key]["TP"] += 1
            elif pred_value is not None and gt_value != pred_value:
                results[key]["FP"] += 1
                results[key]["errors"].append(
                    f"FP: '{pred_value}' statt '{gt_value}' für '{key}'"
                )
            elif pred_value is None and gt_value is not None:
                results[key]["FN"] += 1
                results[key]["errors"].append(
                    f"FN: '{gt_value}' nicht extrahiert für '{key}'"
                )

    return results


def compute_micro_macro_metrics(results):
    """
    Berechnet Mikro- und Makro-Metriken und gibt sie zurück.
    """
    total_TP = sum(metrics["TP"] for metrics in results.values())
    total_FP = sum(metrics["FP"] for metrics in results.values())
    total_FN = sum(metrics["FN"] for metrics in results.values())

    micro_precision = (
        total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    )
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    macro_precision = macro_recall = macro_f1 = 0
    category_count = len(results)

    for key in results:
        TP = results[key]["TP"]
        FP = results[key]["FP"]
        FN = results[key]["FN"]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    macro_precision /= category_count
    macro_recall /= category_count
    macro_f1 /= category_count

    micro_metrics = (micro_precision, micro_recall, micro_f1)
    macro_metrics = (macro_precision, macro_recall, macro_f1)

    return micro_metrics, macro_metrics
    # return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1


def save_evaluation_results(results, micro_metrics, macro_metrics, output_filename):
    """
    Speichert die Evaluationsergebnisse in einer Datei.
    """
    micro_precision, micro_recall, micro_f1 = micro_metrics
    macro_precision, macro_recall, macro_f1 = macro_metrics

    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("=== Kategorie-Ergebnisse ===\n")
        for category, metrics in results.items():
            TP = metrics["TP"]
            FP = metrics["FP"]
            FN = metrics["FN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            file.write(f"\nKategorie: {category}\n")
            file.write(
                f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}\n"
            )
            file.write("Fehler:\n")
            for error in metrics["errors"]:
                file.write(f"  - {error}\n")

        file.write("\n=== Gesamtergebnisse ===\n")
        file.write(
            f"Micro-Precision: {micro_precision:.2f}, Micro-Recall: {micro_recall:.2f}, Micro-F1-Score: {micro_f1:.2f}\n"
        )
        file.write(
            f"Macro-Precision: {macro_precision:.2f}, Macro-Recall: {macro_recall:.2f}, Macro-F1-Score: {macro_f1:.2f}\n"
        )

    print(f"Ergebnisse wurden erfolgreich in '{output_filename}' gespeichert.")


def main():
    # Definiere Pfade und lade Dateien
    base_folder = os.path.dirname(__file__)
    llm_name = "Llama"
    predictions_filename = "meta-llama_Llama-3.2-1B-Instruct_predictions.json"
    ground_truth_filename = "ground_truth.json"

    predictions_file_path = get_predictions_path(
        base_folder, llm_name, predictions_filename
    )
    ground_truth_file_path = os.path.join(base_folder, ground_truth_filename)

    predictions = load_json_file(predictions_file_path)
    ground_truth = load_json_file(ground_truth_file_path)

    results = calculate_metrics(ground_truth, predictions)
    micro_metrics, macro_metrics = compute_micro_macro_metrics(results)

    output_filename = "evaluation_results.txt"
    save_evaluation_results(results, micro_metrics, macro_metrics, output_filename)


if __name__ == "__main__":
    main()
