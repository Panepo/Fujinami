# Prevent pytest from collecting the standalone smoke-test script.
collect_ignore = ["test_docling_serve.py"]


def pytest_addoption(parser):
	parser.addoption(
		"--collection",
		action="store",
		default="S510AD",
		help="RAG collection name used by evaluation tests.",
	)
