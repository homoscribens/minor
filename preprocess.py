import json
import pathlib
from xml.etree import ElementTree as ET

from tqdm import tqdm


def main(eaf_content, output_dir):
    root = ET.fromstring(eaf_content)

    # Re-extracting time slots
    time_slots = {}
    for time_slot in root.findall(".//TIME_SLOT"):
        time_slot_id = time_slot.get("TIME_SLOT_ID")
        time_value = time_slot.get("TIME_VALUE")
        time_slots[time_slot_id] = time_value

    # Re-extracting annotations for user_utterance and sys_utterance
    user_utterances = []
    sys_utterances = []

    for tier in root.findall(".//TIER"):
        tier_id = tier.get("TIER_ID")
        for annotation in tier.findall(".//ANNOTATION"):
            ann_value = annotation.find(".//ANNOTATION_VALUE")
            ref_time_slot = annotation.find(".//ALIGNABLE_ANNOTATION")
            if ann_value is not None and ann_value.text and ref_time_slot is not None:
                annotation_detail = {
                    "utterance": ann_value.text,
                    "start_time": int(time_slots[ref_time_slot.get("TIME_SLOT_REF1")]),
                    "end_time": int(time_slots[ref_time_slot.get("TIME_SLOT_REF2")]),
                }
                if "user_utterance" in tier_id.lower():
                    user_utterances.append(annotation_detail)
                elif "sys_utterance" in tier_id.lower():
                    sys_utterances.append(annotation_detail)

    # Combining user_utterance and sys_utterance annotations into a single list
    combined_utterances = [
        {
            "type": "User",
            "utterance": ann["utterance"],
            "start_time": ann["start_time"],
            "end_time": ann["end_time"],
        }
        for ann in user_utterances
    ] + [
        {
            "type": "System",
            "utterance": ann["utterance"],
            "start_time": ann["start_time"],
            "end_time": ann["end_time"],
        }
        for ann in sys_utterances
    ]

    grouped_utterances = {}
    for utterance in combined_utterances:
        time_key = (utterance["start_time"], utterance["end_time"])
        if time_key not in grouped_utterances:
            grouped_utterances[time_key] = {"System": None, "User": None}
        grouped_utterances[time_key][utterance["type"]] = utterance["utterance"]

    # Convert the grouped utterances to a list format for JSON
    json_output = []
    for time_key, utterances in grouped_utterances.items():
        json_output.append(
            {
                "start_time": time_key[0],
                "end_time": time_key[1],
                "System": utterances["System"],
                "User": utterances["User"],
            }
        )

    # Sorting the JSON data based on start_time
    sorted_json_output = sorted(json_output, key=lambda x: x["start_time"])

    # Writing the sorted data to a new JSON file
    output_sorted_json_path = output_dir / (eaf_path.stem + ".json")
    with open(output_sorted_json_path, "w", encoding="utf-8") as json_file:
        json.dump(sorted_json_output, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_dir = pathlib.Path("./elan")
    output_dir = pathlib.Path("./preprocessed")
    for eaf_path in tqdm(data_dir.glob("*.eaf")):
        with open(eaf_path, "r", encoding="utf-8") as eaf_file:
            eaf_content = eaf_file.read()
        main(eaf_content, output_dir)
