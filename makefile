PIXI_PATH := ~/.pixi/bin/pixi

install:
	curl -fsSL https://pixi.sh/install.sh | bash ; $(PIXI_PATH) install

run:
	$(PIXI_PATH) run python -m book

clean:
	rm -rf .tmp
	rm learncpp.pdf
