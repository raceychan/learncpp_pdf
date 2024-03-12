install:
	curl -fsSL https://pixi.sh/install.sh | bash ; pixi install
run:
	pixi run python -m src.main
