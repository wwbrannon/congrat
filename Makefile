SHELL := /bin/bash

.PHONY: clean pretrain eval score \
        score_gnn_pretrain score_lm_pretrain_causal score_lm_pretrain_masked \
        score_clip_graph_causal score_clip_graph_masked

clean:
	find . -name '__pycache__' -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '*.pyc'       -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name '*.pyo'       -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name '*.egg-info'  -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '*~'          -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name tags          -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name tags.lock     -not -path '*/\.git/*' -exec rm -f {} \+

pretrain:
	@ set -e; \
	for c in configs/lm-pretrain/*/*.yaml; do \
		bin/trainer.py fit -c "$$c"; \
	done
	
	@ set -e; \
	for c in configs/gnn-pretrain*/*/*.yaml; do \
		bin/trainer.py fit -c "$$c"; \
	done
	
	@ set -e; \
	for c in configs/clip-graph{,-directed}/inductive-causal/*/*.yaml; do \
		bin/trainer.py fit -c "$$c"; \
	done
	
	@ set -e; \
	for c in configs/clip-graph{,-directed}/inductive-masked/*/*.yaml; do \
		bin/trainer.py fit -c "$$c"; \
	done

eval:
	bin/eval.py batch -p -r -s test -d cpu --out-dir data/evals/ -f configs/comparisons.yaml

score: score_gnn_pretrain score_lm_pretrain_causal score_lm_pretrain_masked \
       score_clip_graph_causal score_clip_graph_masked

check_defined = \
	$(strip $(foreach 1, $1, $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
	$(if $(value $1), , $(error Undefined $1$(if $2, ($2))))

SCORE_MSG = Must specify the split to score with environment variable \
			SPLIT, acceptable values train, test, val; e.g. \
			SPLIT=val make score

# whether causal or masked for the text component of the eval dataset
# in the gnn-pretrain case doesn't matter, we don't use text at all
score_gnn_pretrain:
	$(call check_defined, SPLIT, ${SCORE_MSG})
	
	@ set -e; \
	for p in lightning_logs/gnn-pretrain/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py gnn_pretrain -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/causal.yaml" \
				-d cuda -s "$(SPLIT)"; \
		done \
	done
	
	@ set -e; \
	for p in lightning_logs/gnn-pretrain-directed/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py gnn_pretrain -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/causal-directed.yaml" \
				-d cuda -s "$(SPLIT)"; \
		done \
	done

# for bin/score.py pretrain_lm, we need to specify the pooling mode and
# normalization behavior. they aren't used in the text pretraining task, but
# are needed to produce these sentence embeddings and are specified in
# clip_graph. for comparability, they should be the same as used in the
# clip-graph models. see the -p and-n options to bin/score.py -- the defaults
# used here without those options are to use mean-pooling and normalization.
score_lm_pretrain_causal:
	$(call check_defined, SPLIT, ${SCORE_MSG})
	
	@ set -e; \
	for p in lightning_logs/lm-pretrain/*; do \
		for v in "$$p"/causal/*; do \
			echo "Scoring $$v..." && \
			bin/score.py lm_pretrain -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/causal.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done

score_lm_pretrain_masked:
	$(call check_defined, SPLIT, ${SCORE_MSG})
	
	@ set -e; \
	for p in lightning_logs/lm-pretrain/*; do \
		for v in "$$p"/masked/*; do \
			echo "Scoring $$v..." && \
			bin/score.py lm_pretrain -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/masked.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done

score_clip_graph_causal:
	$(call check_defined, SPLIT, ${SCORE_MSG})
	
	@ set -e; \
	for p in lightning_logs/clip-graph/inductive-causal/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py clip_graph -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/causal.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done
	
	for p in lightning_logs/clip-graph-directed/inductive-causal/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py clip_graph -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/causal-directed.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done

score_clip_graph_masked:
	$(call check_defined, SPLIT, ${SCORE_MSG})
	
	@ set -e; \
	for p in lightning_logs/clip-graph/inductive-masked/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py clip_graph -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/masked.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done
	
	for p in lightning_logs/clip-graph-directed/inductive-masked/*; do \
		for v in "$$p"/*; do \
			echo "Scoring $$v..." && \
			bin/score.py clip_graph -i "$$v" -o data/embeds/ \
				-c "configs/eval-datasets/$$(basename "$$p")/masked-directed.yaml" \
				-p -d cuda -s "$(SPLIT)"; \
		done \
	done
