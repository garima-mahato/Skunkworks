class ConstrainedBeamSearch(object):
    r"""
    Implements Constrained Beam Search for decoding the most likely sequences conditioned on a
    Finite State Machine with specified state transitions.
    """

    def __init__(
        self,
        eos_token_ids: List[int],
        max_steps: int = 20,
        beam_size: int = 5,
        per_node_beam_size: Optional[int] = None,
        use_hypo: bool = False,
        tokenizer=None,
    ):
        self._eos_token_ids = eos_token_ids
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or self.beam_size
        self.num_keep_best = 1
        self.length_penalty = 1
        self.use_hypo = use_hypo
        self.tokenizer = tokenizer

    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: List[torch.Tensor],
        step: StepFunctionType,
        fsm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # shape: (batch_size, num_fsm_states, num_fsm_states, vocab_size)
        batch_size, num_fsm_states, _, vocab_size = fsm.size()

        # generated hypotheses
        generated_hyps = [
            [BeamHypotheses(self.num_keep_best, self.max_steps, self.length_penalty, early_stopping=False)
            for _ in range(num_fsm_states)]
            for bb in range(batch_size)
        ]

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. None for
        # the first. Stores the index n for the parent prediction.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop because we are going
        # from a single decoder input (the output from the encoder) to the top `beam_size`
        # decoder outputs per FSM state. On the other hand, within the main loop we are going
        # from the `beam_size` elements of the beam (per FSM state) to `beam_size`^2 candidates
        # from which we will select the top `beam_size` elements for the next iteration.

        curr_ids = (
            start_predictions.expand(batch_size, self.beam_size*num_fsm_states)
            .reshape(batch_size*self.beam_size*num_fsm_states, 1)
        )
        # shape: start_class_log_probabilities (batch_size, vocab_size)
        start_class_logits, state = step(curr_ids, start_state)
        start_class_log_probabilities = torch.nn.functional.log_softmax(start_class_logits, dim=-1)
        start_class_log_probabilities = start_class_log_probabilities[:batch_size, :]
        vocab_size = start_class_log_probabilities.size(-1)

        start_state_predictions = start_class_log_probabilities.view(
            batch_size, 1, vocab_size
        ).expand(batch_size, num_fsm_states, vocab_size)

        start_state_predictions = start_state_predictions.masked_fill(
            (1 - fsm[:, 0, :, :]).to(dtype=torch.bool), float("-inf")
        )

        # (batch_size, num_fsm_states, beam_size)
        start_top_log_probabilities, start_predicted_classes = start_state_predictions.topk(
            self.beam_size
        )
        # shape: (batch_size, num_fsm_states, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.full((1, vocab_size), float("-inf")).to(
            start_predictions.device
        )
        log_probs_after_end[:, self._eos_token_ids] = 0.0

        #state = {
            #key: _enlarge_single_tensor(value, batch_size, num_fsm_states, self.beam_size)
            #for (key, value) in state.items()
        #}

        step_state_mask = fsm.view(
            batch_size, num_fsm_states, num_fsm_states, 1, vocab_size
        ).expand(batch_size, num_fsm_states, num_fsm_states, self.beam_size, vocab_size)

        curr_len = curr_ids.shape[1]
        for timestep in range(self.max_steps - curr_len - 1):
            # shape: (batch_size * beam_size * num_fsm_states, )
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size * num_fsm_states
            )
            cur_finished = (last_predictions==self._eos_token_ids[0])
            for eos_token in self._eos_token_ids[1:]:
                cur_finished = (cur_finished | (last_predictions==eos_token))
            if cur_finished.all():
                break

            curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)

            class_logits, state = step(curr_ids, state)
            class_log_probabilities = torch.nn.functional.log_softmax(class_logits, dim=-1)
            #last_predictions_expanded = (
                #last_predictions.view(-1)
                #.unsqueeze(-1)
                #.expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            #)
            cur_finished_expanded = (
                cur_finished.unsqueeze(-1)
                .expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            )

            cleaned_log_probabilities = torch.where(
                #last_predictions_expanded == self._eos_token_ids,
                cur_finished_expanded,
                log_probs_after_end,
                class_log_probabilities,
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, vocab_size
            )

            device = start_predictions.device
            restricted_predicted_classes = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_log_probs = torch.FloatTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_indices = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)

            expanded_last_log_probabilities = last_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, 1
            ).expand(batch_size, num_fsm_states, self.beam_size, self.per_node_beam_size)

            for i in range(num_fsm_states):
                # shape (batch_size, num_fsm_states, self.beam_size, vocab_size)
                state_log_probabilities = cleaned_log_probabilities

                state_log_probabilities = state_log_probabilities.masked_fill(
                    (1 - step_state_mask[:, :, i, :, :]).to(dtype=torch.bool), -1e20
                )
                top_log_probabilities, predicted_classes = state_log_probabilities.topk(
                    self.per_node_beam_size
                )
                summed_top_log_probabilities = (
                    top_log_probabilities + expanded_last_log_probabilities
                )
                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_predicted_classes = predicted_classes.reshape(batch_size, -1)

                if not self.use_hypo:
                    # shape (batch_size, beam_size)
                    state_beam_log_probs, state_beam_indices = reshaped_summed.topk(self.beam_size)
                    # shape (batch_size, beam_size)
                    state_predicted_classes = reshaped_predicted_classes.gather(1, state_beam_indices)
                else:
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_beam_log_probs, candidate_beam_indices = reshaped_summed.topk(
                            self.beam_size*self.per_node_beam_size, sorted=True, largest=True)
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_predicted_classes = reshaped_predicted_classes.gather(1, candidate_beam_indices)
                    next_batch_beam = []
                    for batch_ex in range(batch_size):
                        next_sent_beam = []
                        for word_id, beam_id, log_prob in zip(candidate_predicted_classes[batch_ex],
                                    candidate_beam_indices[batch_ex],
                                    candidate_beam_log_probs[batch_ex]):
                            if word_id.item() in self._eos_token_ids:
                                generated_hyps[batch_ex][i].add(
                                    curr_ids[batch_ex * self.beam_size*num_fsm_states + beam_id/self.per_node_beam_size, :].clone(),
                                    log_prob.item()
                                )
                            else:
                                next_sent_beam.append((word_id, beam_id, log_prob))
                            if len(next_sent_beam) == self.beam_size:
                                break
                        assert len(next_sent_beam) == self.beam_size
                        next_batch_beam.extend(next_sent_beam)
                    state_predicted_classes = torch.tensor([x[0] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_indices = torch.tensor([x[1] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_log_probs = torch.tensor([x[2] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)

                restricted_predicted_classes[:, i, :] = state_predicted_classes
                restricted_beam_indices[:, i, :] = state_beam_indices
                restricted_beam_log_probs[:, i, :] = state_beam_log_probs

            restricted_predicted_classes = restricted_predicted_classes.view(batch_size, -1)
            predictions.append(restricted_predicted_classes)

            backpointer = restricted_beam_indices / self.per_node_beam_size
            backpointers.append(backpointer.view(batch_size, -1))

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, num_fsm_states, -1)

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, num_fsm_states * self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, num_fsm_states * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return (
                    state_tensor.reshape(batch_size, num_fsm_states * self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * num_fsm_states * self.beam_size, *last_dims)
                )
            # reorder states
            if state is not None:
                state = tuple(track_back_state(value) for value in state)
            curr_ids = track_back_state(curr_ids)

        last_predictions = predictions[-1].reshape(
            batch_size * self.beam_size * num_fsm_states
        )
        curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_predictions = all_predictions.view(batch_size, num_fsm_states, self.beam_size, -1)
        assert (all_predictions == curr_ids.reshape(batch_size, num_fsm_states,
                self.beam_size, -1)[:,:,:,1:]).all()

        if self.use_hypo:
            decoded = all_predictions.new(batch_size, num_fsm_states, 1,
                    self.max_steps).fill_(self._eos_token_ids[0])
            scores = last_log_probabilities.new(batch_size, num_fsm_states,
                    1).fill_(-1e5)
            for batch_ex in range(batch_size):
                for i in range(num_fsm_states):
                    beam = all_predictions[batch_ex, i, 0, :]
                    log_prob = last_log_probabilities[batch_ex, i, 0]
                    generated_hyps[batch_ex][i].add(
                        beam.clone(),
                        log_prob.item()
                    )
                    hyps = generated_hyps[batch_ex][i].hyp
                    assert len(hyps) == 1
                    score, sent = hyps[0]
                    decoded[batch_ex, i, 0, :len(sent)] = sent
                    scores[batch_ex, i, 0] = score
            all_predictions = decoded
            last_log_probabilities = scores

        # pad to the same length, otherwise DataParallel will give error
        pad_len = self.max_steps - all_predictions.shape[-1]
        if pad_len > 0:
            padding_ids = all_predictions.new(
                    batch_size, num_fsm_states, self.beam_size,
                    pad_len).fill_(self._eos_token_ids[0])
            all_predictions = torch.cat([all_predictions, padding_ids], dim=-1)

        return all_predictions, last_log_probabilities
