import unittest
import json
from src.synth_dialogues import lmflow_training_format, dataset2sharegpt

class TestDialogueTransformations(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {
                "script": "Test patient script",
                "dialogue": [
                    {"role": "doctor", "content": "Hello, how can I help you?"},
                    {"role": "patient", "content": "I have a headache"}
                ]
            }
        ]

    def test_lmflow_format(self):
        result = lmflow_training_format(self.sample_data)
        self.assertEqual(result["type"], "text_only")
        self.assertTrue("instances" in result)
        self.assertTrue(len(result["instances"]) > 0)
        self.assertTrue("text" in result["instances"][0])

    def test_sharegpt_format(self):
        input_data = json.dumps(self.sample_data[0]) + "\n"
        with open("test_input.jsonl", "w") as f:
            f.write(input_data)
        
        dataset2sharegpt("test_input.jsonl", "test_output.json")
        
        with open("test_output.json") as f:
            result = json.load(f)
        
        self.assertTrue("conversations" in result)
        self.assertTrue(len(result["conversations"]) > 0)
        self.assertEqual(result["conversations"][0][0]["from"], "system")
        self.assertEqual(result["conversations"][0][1]["from"], "user")
        self.assertEqual(result["conversations"][0][2]["from"], "assistant")

if __name__ == '__main__':
    unittest.main()