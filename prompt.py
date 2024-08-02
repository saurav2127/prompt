import argparse
import pandas as pd
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--input_file", type=str, default="test_csv.csv")
    parser.add_argument("--output_file", type=str, default="generated_responses.csv")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to process")
    return parser.parse_args()

# Function to generate responses using the specified LLM model
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to create the initial prompt for generating responses
def create_prompt(system_prompt, context):
    prompt = f'''
[System] {system_prompt}
[Context] {context}
[Response] '''
    return prompt

# System prompt based on your provided example
system_prompt = '''We are performing argumentation-based negotiation. We are in a hotel-booking scenario.

Examples:
1. Context: "[SOC] [Agent] "Welcome! It's so exciting to be planning your trip with you.  To help me find the perfect tour package for you, could you tell me a little about what you're looking for?  Are you dreaming of a relaxing beach vacation, an adventure-filled journey, a cultural immersion, or something else entirely?" 
 [Traveler] "I'm definitely the type who's always ready to stop for that landmark, event, scenic vista or attraction.  I don't want to miss anything!  So my ideal tour package would be packed with stops at all the must-see places, whether it's a famous historical building, a bustling festival, a breathtaking mountain view, or a unique local attraction.  I want to experience it all, and I'm not afraid to go off the beaten path to find something truly special." 
 [Agent] "The 'Sight Seeker' package is perfect for those who want to experience the best of both worlds – adventure and relaxation.  It's priced at $53,462 and includes cozy cottage accommodations, fine dining, and private transportation. You'll enjoy immersive virtual reality experiences, exciting outdoor activities, and a guided photography tour to capture those breathtaking moments. It's a truly unforgettable experience!" 
 [Traveler] "I appreciate the package you've put together, but $53,462 is a bit outside my budget.  Would you be willing to work with me on a price closer to $37,651? I'm sure we can find a way to make this work for both of us." 
 [EOC]"
Response: "[SOR] [Agent] "This package is carefully crafted to provide you with an unforgettable experience, and the price of $53,462 reflects the high quality of the accommodations, activities, and itinerary. While I understand you might be looking for a lower price, I want to ensure that we are delivering exceptional value for your investment. This price is non-negotiable as it includes all the elements that make this tour truly special. However, I'm happy to discuss your specific needs and see if we can adjust the itinerary in a way that might fit your budget better. Would you like to discuss that?" 
 [EOR]"

2. Context: "[SOC] [Agent] "For the ultimate shopping experience, I recommend our 'Shopping Shark' package! It's priced at $53,681 and includes a cozy cottage for your stay, fine dining, and private transportation.  But that's not all! You'll also enjoy the convenience of shipping and packing assistance, a photography tour to capture your amazing finds, and a personal assistant to help you navigate the shops.  Ready to unleash your inner shopping shark?" 
 [Traveler] "That's a great package, but $53,681 is a bit out of my budget. Would you be willing to consider $38,830? I'm flexible and open to discussing other options to get closer to that price point." 
 [Agent] "I appreciate you working with us on this, and I understand you're looking for a great deal. However, $38,830 would be difficult for us to offer given the inclusions in this package. To ensure we can deliver the high-quality experience you deserve, I'm proposing a counteroffer of $52,763.02. This price accounts for the premium flights, accommodations, and unique activities included in this package." 
 [Traveler] "I'm actually not interested in the shipping and packing assistance included in the package. Could we remove that from the tour and adjust the price accordingly?" 
 [EOC]"
Response: "[SOR] [Agent] "Okay, so we can definitely remove the shipping and packing assistance from the package. That will bring the total cost down to **$47,831.02**.  Does that work for you?" 
Here's why I made these changes:

- **Round the number:** It's highly unlikely a travel package would be priced to the hundredth of a cent. Rounding to the nearest cent makes the price more realistic and easier to understand.
- **Use commas:** For larger numbers, commas improve readability.
- **Use a friendly tone:** A travel agent should be friendly and helpful, so using phrases like "Okay, so..." and "Does that work for you?" create a more conversational and welcoming tone.

Remember, the goal is to be clear, concise, and friendly in your communication. 
 [EOR]"

3. Context: "[SOC] [EOC]"
Response: "[SOR] [Agent] "Welcome! It's so exciting to be planning your trip with you.  To help me find the perfect tour package for you, could you tell me a little about what you're looking for?  Are you dreaming of a relaxing beach vacation, an adventure-filled journey, a cultural immersion, or something else entirely?" 
 [EOR]"

Instructions:

1. For each context, identify the next speaker and generate a response from them.
2. Ensure that the response is relevant and logical within the negotiation context.
3. If the context ends with [EOC], start the response with an Agent's dialogue.
4. Make sure there are no repeated responses.
5. Ensure the responses follow the negotiation and friendly tone as demonstrated in the examples.'''

# Function to determine the next speaker based on the last speaker in the context
def get_next_speaker(context):
    if "[Agent]" in context.split("[EOC]")[-1]:
        return "[Traveler]"
    else:
        return "[Agent]"

def main():
    args = get_args()
    set_seed(42)  # Ensure reproducibility

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_4bit=True, device_map="auto"
    )

    # Read the input CSV file
    data = pd.read_csv(args.input_file)

    # Limit to the specified number of samples
    data = data.sample(n=args.num_samples, random_state=42)

    # Initialize a list to store the results
    results = []

    # Iterate over each row in the DataFrame
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        conv_id = row['conv_id']
        context = row['context']
        response = row['response']
        
        # Determine the next speaker
        next_speaker = get_next_speaker(context)
        
        # Create the prompt using system prompt and current context
        prompt = create_prompt(system_prompt, context)
        
        try:
            # Get completion from the specified model
            generated_response = generate_response(model, tokenizer, prompt)
            
            # Format the generated response
            generated_response = f"[SOR] {next_speaker} \"{generated_response.strip()}\" [EOR]"
            
            # Append the result to the list
            results.append([conv_id, context, response, generated_response])
            
            if i % 10 == 0:
                print(i + 1, 'done')

        except Exception as e:
            print(repr(e))
            time.sleep(20)

    # Create a DataFrame from the results
    output_df = pd.DataFrame(results, columns=['conv_id', 'context', 'response', 'generated_response'])

    # Save the DataFrame to the output CSV file
    output_df.to_csv(args.output_file, index=False)

    print("Response generation completed and written to", args.output_file)

if __name__ == "__main__":
    main()
