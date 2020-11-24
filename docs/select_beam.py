def select_best_beam_with_constraints(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
    given_constraints: torch.Tensor,
    min_constraints_to_satisfy: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    batch_size, num_states, beam_size, max_decoding_steps = beams.size()

    best_beams: List[torch.Tensor] = []
    best_beam_log_probabilities: List[torch.Tensor] = []

    for i in range(batch_size):
        # fmt: off
        valid_states = [
            s for s in range(2 ** given_constraints[i].item())
            if bin(s).count("1") >= min(given_constraints[i], min_constraints_to_satisfy)
        ]
        # fmt: on

        valid_beams = beams[i, valid_states, 0, :]
        valid_beam_log_probabilities = beam_log_probabilities[i, valid_states, 0]

        selected_index = torch.argmax(valid_beam_log_probabilities)
        best_beams.append(valid_beams[selected_index, :])
        best_beam_log_probabilities.append(valid_beam_log_probabilities[selected_index])

    # shape: (batch_size, max_decoding_steps)
    return (torch.stack(best_beams).long().to(beams.device),
            torch.stack(best_beam_log_probabilities).to(beams.device))
