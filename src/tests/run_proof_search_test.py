import subprocess
import unittest
import os

class TestProofWalaSearchCommand(unittest.TestCase):
    def test_proof_wala_search_command(self):
        """
        Test that the 'proof-wala-search' command runs successfully with the given configuration.
        """
        # Construct the command as a single string.
        command = (
            "proof-wala-search --config-dir=src/proof_wala/main/config "
            "--config-name=eval_simple_lean_test_multilingual_easy.yaml"
        )

        try:
            # Run the command using shell=True so that the shell does the PATH lookup.
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=700
            )
        except subprocess.TimeoutExpired as e:
            self.fail(f"'proof-wala-search' command timed out: {e}")
        except Exception as e:
            self.fail(f"Error running 'proof-wala-search': {e}")

        # Check that the command exited with a return code of 0.
        self.assertEqual(
            result.returncode, 0,
            msg=f"'proof-wala-search' failed with return code {result.returncode}. Stderr: {result.stderr}"
        )

        # Print all the files in the .log/proofs_logs/simple_lean_test_easy/simple_lean_test
        # directory to see what was generated.
        # Do a list and pick the last folder in the list as per the sorted order
        dirs = sorted(os.listdir(".log/proofs_logs/simple_lean_test_easy/simple_lean_test"))
        print(dirs)
        last_dir = dirs[-1]
        gpu_dir = os.path.join(".log/proofs_logs/simple_lean_test_easy/simple_lean_test", last_dir, "gpu_None_0")
        log_file = os.path.join(gpu_dir, "eval_dataset_multiple_attempts.log")
        print("Log file:", log_file)
        with open(log_file, "r") as f:
            print(f.read())

if __name__ == '__main__':
    unittest.main()
