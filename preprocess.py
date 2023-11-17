import pathlib
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm


def preprocess_eaf(eaf_path):
    with open(eaf_path, "r", encoding="utf-8") as eaf_file:
        eaf_content = eaf_file.read()

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
            if (
                ann_value is not None
                and ann_value.text
                and ref_time_slot is not None
            ):
                annotation_detail = {
                    "utterance": ann_value.text,
                    "start_time": int(
                        time_slots[ref_time_slot.get("TIME_SLOT_REF1")]
                    ),
                    "end_time": int(
                        time_slots[ref_time_slot.get("TIME_SLOT_REF2")]
                    ),
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
        grouped_utterances[time_key][utterance["type"]] = utterance[
            "utterance"
        ]

    # Convert the grouped utterances to a list format for JSON
    eaf_output = []
    for time_key, utterances in grouped_utterances.items():
        eaf_output.append(
            {
                "start_time": time_key[0],
                "end_time": time_key[1],
                "System": utterances["System"],
                "User": utterances["User"],
            }
        )

    # Sorting the data based on start_time
    sorted_eaf_output = sorted(eaf_output, key=lambda x: x["start_time"])

    return sorted_eaf_output


def preprocess_dump(eaf_path, sorted_eaf_output):
    dump_path = pathlib.Path("./dumpfiles")
    dump_file = dump_path / (eaf_path.stem + ".csv")

    if not dump_file.exists():
        return None
    dump_df = pd.read_csv(dump_file).filter(regex=r"^(?!word#)")

    user = []
    system = []
    for i, dump_start_time in enumerate(dump_df["start(exchange)[ms]"]):
        for eaf in sorted_eaf_output:
            eaf_start_time = eaf["start_time"]
            if dump_start_time - eaf_start_time < 5:
                user.append(eaf["User"])
                system.append(eaf["System"])
                break

    assert len(dump_df) == len(system)

    dump_df["system_utterance"] = system
    dump_df["user_utterance"] = user

    return dump_df


def main(eaf_path, output_dir):
    preprocessed_data = preprocess_eaf(eaf_path)
    dump_df = preprocess_dump(eaf_path, preprocessed_data)

    if dump_df is not None:
        dump_df.to_csv(output_dir / (eaf_path.stem + ".csv"), index=False)


if __name__ == "__main__":
    data_dir = pathlib.Path("./elan")
    output_dir = pathlib.Path("./preprocessed")
    if not output_dir.exists():
        output_dir.mkdir()
    for eaf_path in tqdm(data_dir.glob("*.eaf")):
        main(eaf_path, output_dir)
