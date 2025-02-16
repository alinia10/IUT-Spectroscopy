import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

from preprocess import preprocess  # Ensure this is the correct import for your processing function
from conf import config  # Ensure this is the correct import for your configuration

# Replace with your bot's API token
TOKEN = config["api"]["telegram_token"]  # Or set it directly: TOKEN = "YOUR_BOT_TOKEN"

# Ensure DATA and results directories exist
if not os.path.exists('./DATA'):
    os.makedirs('./DATA')
if not os.path.exists('./results'):
    os.makedirs('./results')

# Function to handle the /start command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Please send me a link to a ZIP file.')

# Function to download a file from a URL
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    else:
        return False

# Function to handle the received link
async def handle_link(update: Update, context: CallbackContext) -> None:
    try:
        # Get the link from the user's message
        link = update.message.text

        # Validate the link
        if not link.endswith('.zip'):
            await update.message.reply_text('Please send a valid link to a ZIP file.')
            return

        # Download the file
        file_name = os.path.basename(link)
        file_path = f"./DATA/{file_name}"
        if not download_file(link, file_path):
            await update.message.reply_text('Failed to download the file. Please check the link and try again.')
            return

        # Process the file using the main function from preprocess.py
        preprocess(file_path)  # Pass the file path to the main function

        # Send all files in the ./results/ directory
        results_dir = './results'
        for result_file_name in os.listdir(results_dir):
            result_file_path = os.path.join(results_dir, result_file_name)
            if os.path.isfile(result_file_path):  # Ensure it's a file
                with open(result_file_path, 'rb') as result_file:
                    await update.message.reply_document(document=result_file)

        # Clean up
        os.remove("./DATA")  # Remove the downloaded ZIP file
        for result_file_name in os.listdir(results_dir):
            result_file_path = os.path.join(results_dir, result_file_name)
            if os.path.isfile(result_file_path):
                os.remove(result_file_path)  # Remove each result file

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

def main() -> None:
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Register the command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()