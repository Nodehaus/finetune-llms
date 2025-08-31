import torch
import wandb

from .evaluation_questions import format_question, get_evaluation_dataset


def evaluate_model(model, tokenizer, log_to_wandb=True):
    """
    Evaluate the model on trademark classification questions.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        log_to_wandb: Whether to log results to wandb

    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    questions = get_evaluation_dataset()

    correct_answers = 0
    total_questions = len(questions)
    results = []

    print(f"Starting evaluation on {total_questions} questions...")

    with torch.no_grad():
        for i, question_data in enumerate(questions):
            # Format the prompt
            prompt = format_question(question_data, include_examples=True)

            # Tokenize input
            inputs = tokenizer(
                [prompt], return_tensors="pt", max_length=2048, truncation=True
            ).to(model.device)

            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Short answer expected
                temperature=0.1,  # Low temperature for more deterministic answers
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer (everything after the last "A:")
            answer_start = generated_text.rfind("A:")
            if answer_start != -1:
                generated_answer = generated_text[answer_start + 2 :].strip().lower()

                # Clean up the generated answer - take only the first word
                generated_answer = (
                    generated_answer.split()[0] if generated_answer.split() else ""
                )

                # Remove any punctuation
                generated_answer = generated_answer.strip(".,!?;:")

            else:
                generated_answer = ""

            # Check if answer is correct
            correct_answer = question_data["answer"].lower()
            is_correct = generated_answer == correct_answer

            if is_correct:
                correct_answers += 1

            # Store result for logging
            result = {
                "question": question_data["question"],
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "full_response": generated_text[len(prompt) :].strip(),
            }
            results.append(result)

            # Print progress every 10 questions
            if (i + 1) % 10 == 0:
                print(
                    f"Processed {i + 1}/{total_questions} questions. Current accuracy: {correct_answers / (i + 1):.3f}"
                )

    # Calculate final accuracy
    accuracy = correct_answers / total_questions

    # Prepare metrics
    metrics = {
        "eval_accuracy": accuracy,
        "eval_correct_answers": correct_answers,
        "eval_total_questions": total_questions,
        "eval_results": results,
    }

    print(f"\nEvaluation completed!")
    print(f"Accuracy: {accuracy:.3f} ({correct_answers}/{total_questions})")

    # Log to wandb if requested
    if log_to_wandb and wandb.run is not None:
        # Log main metrics
        wandb.log(
            {
                "eval/accuracy": accuracy,
                "eval/correct_answers": correct_answers,
                "eval/total_questions": total_questions,
            }
        )

        # Create a table for detailed results
        eval_table = wandb.Table(
            columns=[
                "Question",
                "Correct Answer",
                "Generated Answer",
                "Is Correct",
                "Full Response",
            ],
            data=[
                [
                    result["question"],
                    result["correct_answer"],
                    result["generated_answer"],
                    result["is_correct"],
                    result["full_response"][:200] + "..."
                    if len(result["full_response"]) > 200
                    else result["full_response"],
                ]
                for result in results[:50]
            ],  # Limit to first 50 for readability
        )

        wandb.log({"eval/detailed_results": eval_table})

        # Log answer distribution
        answer_counts = {}
        for result in results:
            answer = result["generated_answer"]
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        wandb.log({"eval/answer_distribution": answer_counts})

    return metrics


def custom_eval_callback(model, tokenizer):
    """
    Callback function that can be used during training for periodic evaluation.
    """

    def eval_function():
        return evaluate_model(model, tokenizer, log_to_wandb=True)

    return eval_function
