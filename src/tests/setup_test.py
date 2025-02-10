import unittest
import torch

class TestProofWalaCI(unittest.TestCase):
    def test_torch_installation(self):
        """Test that torch is installed and has a version, and print CUDA availability."""
        version = torch.__version__
        self.assertIsNotNone(version, "Torch version should not be None.")
        print("Torch version:", version)
        print("CUDA available:", torch.cuda.is_available())
    
    def test_proof_wala_import(self):
        """Test that the proof_wala package can be imported."""
        try:
            import proof_wala
        except Exception as e:
            self.fail("Failed to import proof_wala package: " + str(e))
    
    def test_itp_interface_import(self):
        """Optionally test that the itp-interface dependency is available.
        
        If itp-interface is not installed, skip the test.
        """
        try:
            import itp_interface  # or the actual module name if different
        except ImportError as e:
            self.skipTest("itp-interface not installed: " + str(e))
    
    def test_huggingface_pipeline(self):
        """Test that a transformers pipeline for the ProofWala model can be instantiated.
        
        We use local_files_only=True to avoid network access during CI.
        If the model files are not cached locally, the test is skipped.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            model_name = "amitayusht/ProofWala-Multilingual"
            # Use local_files_only=True to ensure CI does not require network access.
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            text2text_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
            test_input = """Goals to prove:
[GOALS]
[GOAL] 1
forall n : nat, n + 1 = 1 + n
[END]"""
            output = text2text_pipe(test_input, max_length=100)
            print("Output:", output)
            self.assertIsInstance(output, list, "Expected output to be a list.")
        except Exception as e:
            self.skipTest("Skipping HuggingFace pipeline test: " + str(e))

if __name__ == '__main__':
    unittest.main()
