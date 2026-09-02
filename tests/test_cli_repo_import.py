import unittest
from pathlib import Path


class TestCLIRepoImport(unittest.TestCase):
    def test_public_cli_prepend_repo_root_before_package_import(self):
        scripts = [
            Path("tools/train.py"),
            Path("tools/eval_potsdam.py"),
            Path("tools/eval_vaihingen.py"),
            Path("tools/eval_loveda.py"),
        ]

        for script in scripts:
            with self.subTest(script=str(script)):
                text = script.read_text(encoding="utf-8")

                bootstrap = 'sys.path.insert(0, str(REPO_ROOT))'
                package_import = "from sufh_rl"

                self.assertIn(bootstrap, text)
                self.assertIn(package_import, text)

                self.assertLess(
                    text.index(bootstrap),
                    text.index(package_import),
                    msg=f"{script}: repository bootstrap must occur before sufh_rl import",
                )


if __name__ == "__main__":
    unittest.main()
