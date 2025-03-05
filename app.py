from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
from google.cloud import translate_v2 as translate
from gtts import gTTS
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from pathlib import Path
import io
import wave
import scipy.io.wavfile as wavfile
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle

# Load environment variables from .env file
load_dotenv('api.env')

app = Flask(__name__)
CORS(app)

# Initialize paths
BASE_DIR = Path("data")
CONVERSATIONS_DIR = BASE_DIR / "conversations"
TRANSLATIONS_DIR = BASE_DIR / "translations"
SUMMARIES_DIR = BASE_DIR / "summaries"

# Create necessary directories
for directory in [BASE_DIR, CONVERSATIONS_DIR, TRANSLATIONS_DIR, SUMMARIES_DIR]:
    directory.mkdir(exist_ok=True)

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_credentials():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_535201760510-8thq35fisfododotdfmevmfujknqop0m.apps.googleusercontent.com (3).json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

class RealEstateAssistant:
    def __init__(self):
        self.translate_client = translate.Client()
        
        # Initialize Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Sync properties from CSV to JSON
        self.sync_properties()
        
        # Load users database with proper columns
        self.users_df = pd.read_csv('users.csv') if os.path.exists('users.csv') else pd.DataFrame(
            columns=['user_id', 'name', 'email', 'phone', 'created_at']
        )
        
        # Load clients database
        self.clients_df = pd.read_csv('clients.csv') if os.path.exists('clients.csv') else pd.DataFrame(
            columns=['client_id', 'user_id', 'client_name', 'status', 'purchase_property', 'purchase_date']
        )

        # Track active conversations
        self.active_conversations = {}

        # Language codes for speech recognition
        self.language_codes = {
            'hi-IN': 'hi-IN',  # Hindi
            'ta-IN': 'ta-IN',  # Tamil
            'te-IN': 'te-IN',  # Telugu
            'kn-IN': 'kn-IN',  # Kannada
            'ml-IN': 'ml-IN'   # Malayalam
        }

    def sync_properties(self):
        """Sync properties from CSV to JSON format"""
        try:
            # First load or create properties DataFrame from CSV
            if os.path.exists('properties.csv'):
                self.properties_df = pd.read_csv('properties.csv')
            else:
                self.properties_df = pd.DataFrame(
                    columns=['id', 'type', 'location', 'price', 'size', 'latitude', 'longitude', 'status', 'amenities']
                )
                # Add some default properties with amenities
                default_properties = [
                    {
                        'id': 'PROP001',
                        'type': 'Apartment',
                        'location': 'Whitefield, Bangalore',
                        'price': 8500000,
                        'size': 1200,
                        'latitude': 12.9698,
                        'longitude': 77.7500,
                        'status': 'available',
                        'amenities': 'Swimming Pool,Gym,24x7 Security,Power Backup'
                    },
                    {
                        'id': 'PROP002',
                        'type': 'Villa',
                        'location': 'HSR Layout, Bangalore',
                        'price': 15000000,
                        'size': 2400,
                        'latitude': 12.9116,
                        'longitude': 77.6474,
                        'status': 'available',
                        'amenities': 'Private Garden,Modular Kitchen,3 Car Parking,Smart Home'
                    }
                ]
                self.properties_df = pd.DataFrame(default_properties)
                self.properties_df.to_csv('properties.csv', index=False)

            # Convert DataFrame to JSON format
            properties_json = []
            for _, prop in self.properties_df.iterrows():
                if prop['status'] == 'available':  # Only include available properties in JSON
                    # Convert amenities string to list
                    amenities_list = prop['amenities'].split(',') if isinstance(prop['amenities'], str) else []
                    
                    property_data = {
                        'id': str(prop['id']),
                        'type': str(prop['type']),
                        'location': str(prop['location']),
                        'price': int(prop['price']),
                        'area': int(prop['size']),
                        'coordinates': {
                            'lat': float(prop['latitude']),
                            'lng': float(prop['longitude'])
                        },
                        'amenities': amenities_list
                    }
                    properties_json.append(property_data)

            # Ensure properties directory exists
            properties_dir = os.path.join(BASE_DIR, 'properties')
            os.makedirs(properties_dir, exist_ok=True)

            # Save to JSON file
            properties_file = os.path.join(properties_dir, 'properties.json')
            with open(properties_file, 'w') as f:
                json.dump(properties_json, f, indent=2)

        except Exception as e:
            print(f"Error syncing properties: {str(e)}")
            # Create empty DataFrame if sync fails
            self.properties_df = pd.DataFrame(
                columns=['id', 'type', 'location', 'price', 'size', 'latitude', 'longitude', 'status', 'amenities']
            )

    def get_all_users(self):
        """Get list of all users"""
        return self.users_df.to_dict('records')

    def get_user_details(self, user_id):
        """Get details of a specific user"""
        user = self.users_df[self.users_df['user_id'] == user_id]
        if len(user) > 0:
            return user.iloc[0].to_dict()
        return None

    def get_user_clients(self, user_id):
        """Get all clients associated with a user"""
        try:
            # First check if the user exists
            user = self.get_user_details(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")

            # Ensure clients.csv exists and has the correct columns
            if not os.path.exists('clients.csv'):
                print("Creating new clients.csv file...")
                self.clients_df = pd.DataFrame(
                    columns=['client_id', 'user_id', 'client_name', 'status', 'purchase_property', 'purchase_date']
                )
                self.clients_df.to_csv('clients.csv', index=False)
            else:
                # Reload the clients DataFrame to ensure we have the latest data
                self.clients_df = pd.read_csv('clients.csv')

            # Get all clients for the user that are not closed
            clients = self.clients_df[
                (self.clients_df['user_id'] == user_id) & 
                (self.clients_df['status'] != 'closed')
            ]

            # Convert to records and ensure all fields are present
            client_records = []
            for _, client in clients.iterrows():
                client_record = {
                    'client_id': str(client['client_id']),  # Ensure client_id is a string
                    'user_id': str(client['user_id']),      # Ensure user_id is a string
                    'client_name': str(client['client_name']),
                    'status': str(client['status']).lower(),
                    'purchase_property': str(client['purchase_property']) if pd.notna(client['purchase_property']) else 'none',
                    'purchase_date': str(client['purchase_date']) if pd.notna(client['purchase_date']) else None
                }
                client_records.append(client_record)

            print(f"Found {len(client_records)} active clients for user {user_id}")
            return client_records

        except Exception as e:
            print(f"Error getting clients for user {user_id}: {str(e)}")
            raise ValueError(f"Error getting clients: {str(e)}")

    def create_client(self, user_id, client_name):
        """Create a new client and associate it with a user"""
        try:
            # Generate a unique client ID
            client_id = f"CLIENT_{len(self.clients_df) + 1}"
            
            # Create new client entry
            new_client = {
                'client_id': client_id,
                'user_id': user_id,
                'client_name': client_name,
                'status': 'active',
                'purchase_property': 'none',
                'purchase_date': None
            }
            
            # Add to DataFrame
            self.clients_df = pd.concat([self.clients_df, pd.DataFrame([new_client])], ignore_index=True)
            
            # Save to CSV
            self.clients_df.to_csv('clients.csv', index=False)
            
            return new_client
        except Exception as e:
            print(f"Error creating client: {e}")
            raise

    def mark_client_done(self, client_id, property_id=None):
        """Mark a client as done and update their status"""
        try:
            # Find the client in the DataFrame
            client_mask = self.clients_df['client_id'] == client_id
            if not client_mask.any():
                print(f"Client {client_id} not found")
                return False
            
            # Update the client's status
            self.clients_df.loc[client_mask, 'status'] = 'closed'
            self.clients_df.loc[client_mask, 'purchase_property'] = property_id if property_id else 'none'
            self.clients_df.loc[client_mask, 'purchase_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # If property was purchased, mark it as sold
            if property_id:
                property_mask = self.properties_df['id'] == property_id
                if property_mask.any():
                    self.properties_df.loc[property_mask, 'status'] = 'sold'
                    # Save properties DataFrame
                    self.properties_df.to_csv('properties.csv', index=False)
            
            # Save clients DataFrame
            self.clients_df.to_csv('clients.csv', index=False)
            return True
        except Exception as e:
            print(f"Error marking client as done: {e}")
            return False

    def get_active_conversation_file(self, client_id):
        """Get or create the current conversation file for a client"""
        # Create client directory if it doesn't exist
        client_dir = CONVERSATIONS_DIR / client_id
        client_dir.mkdir(exist_ok=True)
        
        if client_id not in self.active_conversations:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = client_dir / f"conversation_{timestamp}.json"
            self.active_conversations[client_id] = {
                "filename": filename,
                "data": {
                    "client_id": client_id,
                    "start_time": datetime.now().isoformat(),
                    "status": "active",
                    "conversation": []
                }
            }
        return self.active_conversations[client_id]

    def transcribe_audio(self, audio_data, sample_rate, language):
        try:
            # Convert float32 array to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create in-memory WAV file
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Rewind buffer to start
            wav_buffer.seek(0)
            
            # Use speech recognition
            with sr.AudioFile(wav_buffer) as source:
                audio = self.recognizer.record(source)
                lang_code = self.language_codes.get(language, 'en-IN')
                text = self.recognizer.recognize_google(audio, language=lang_code)
                return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with the speech recognition service; {str(e)}"
        except Exception as e:
            return f"Error processing audio: {str(e)}"

    def text_to_speech(self, text, language):
        try:
            # Extract base language code for gTTS (it doesn't use the -IN suffix)
            base_lang = language.split('-')[0] if '-' in language else language
            tts = gTTS(text=text, lang=base_lang)
            
            # Save to in-memory buffer
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            return mp3_buffer
        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            return None

    def translate_text(self, text, target_language='en'):
        try:
            # Extract base language code for translation (it doesn't use the -IN suffix)
            source_lang = target_language.split('-')[0] if '-' in target_language else target_language
            result = self.translate_client.translate(text, target_language=source_lang)
            return result['translatedText']
        except Exception as e:
            return str(e)

    def store_conversation(self, client_id, new_message):
        """Add new message to the active conversation"""
        try:
            print(f"Storing new message for client: {client_id}")
            conv_data = self.get_active_conversation_file(client_id)
            
            # Add the new message with timestamp
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "original_text": new_message["text"],
                "translated_text": new_message["translated"],
                "language": new_message.get("language", "unknown"),
                "speaker": "client"  # Add speaker identification
            }
            
            conv_data["data"]["conversation"].append(message_data)
            
            # Ensure the client directory exists
            client_dir = CONVERSATIONS_DIR / client_id
            client_dir.mkdir(exist_ok=True)
            
            print(f"Saving conversation to file: {conv_data['filename']}")
            # Save to file
            with open(conv_data["filename"], 'w', encoding='utf-8') as f:
                json.dump(conv_data["data"], f, ensure_ascii=False, indent=2)
            
            print(f"Successfully stored message for client {client_id}")
            return True
        except Exception as e:
            print(f"Error storing conversation for client {client_id}: {e}")
            raise

    def end_conversation(self, client_id):
        """Mark conversation as complete and close the file"""
        if client_id in self.active_conversations:
            conv_data = self.active_conversations[client_id]
            conv_data["data"]["status"] = "completed"
            conv_data["data"]["end_time"] = datetime.now().isoformat()
            
            # Calculate conversation duration
            start_time = datetime.fromisoformat(conv_data["data"]["start_time"])
            end_time = datetime.fromisoformat(conv_data["data"]["end_time"])
            duration = (end_time - start_time).total_seconds()
            conv_data["data"]["duration_seconds"] = duration
            
            # Save final version
            with open(conv_data["filename"], 'w', encoding='utf-8') as f:
                json.dump(conv_data["data"], f, ensure_ascii=False, indent=2)
            
            # Remove from active conversations
            del self.active_conversations[client_id]
            return True
        return False

    def get_all_client_conversations(self, client_id):
        """Get all conversations for a client, including active and completed ones"""
        client_dir = CONVERSATIONS_DIR / client_id
        all_conversations = []
        
        print(f"Searching for conversations for client: {client_id}")
        print(f"Client directory path: {client_dir}")
        
        # Get completed conversations
        if client_dir.exists():
            print(f"Client directory exists. Searching for conversation files...")
            conversation_files = list(client_dir.glob("conversation_*.json"))
            print(f"Found {len(conversation_files)} conversation files")
            
            for conv_file in conversation_files:
                try:
                    print(f"Reading conversation file: {conv_file}")
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        all_conversations.append(conversation)
                        print(f"Successfully loaded conversation from {conv_file}")
                except Exception as e:
                    print(f"Error reading conversation file {conv_file}: {e}")
        else:
            print(f"Client directory does not exist at: {client_dir}")
        
        # Add active conversation if exists
        if client_id in self.active_conversations:
            print(f"Found active conversation for client {client_id}")
            all_conversations.append(self.active_conversations[client_id]["data"])
        else:
            print(f"No active conversation found for client {client_id}")
        
        # Sort conversations by start time
        all_conversations.sort(key=lambda x: x.get("start_time", ""))
        print(f"Total conversations found: {len(all_conversations)}")
        return all_conversations

    def format_summary_response(self, text):
        """Format the summary response to be more readable"""
        # Remove excessive asterisks and hashes
        text = text.replace('**', '')
        text = text.replace('##', '')
        
        # Split into sections
        sections = text.split('\n')
        formatted_sections = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Format headers
            if ':' in section and not section.startswith('*'):
                header, content = section.split(':', 1)
                formatted_sections.append(f"\n=== {header.strip().upper()} ===")
                formatted_sections.append(content.strip())
            # Format bullet points
            elif section.startswith('*'):
                formatted_sections.append(f"\n  • {section[1:].strip()}")
            # Format numbered points
            elif any(section.startswith(str(i)) for i in range(1, 6)):
                formatted_sections.append(f"\n{section}")
            else:
                formatted_sections.append(section)
        
        return '\n'.join(formatted_sections)

    def get_conversation_summary(self, client_id):
        """Get summary of all conversations for a client"""
        try:
            print(f"\nGenerating summary for client: {client_id}")
            all_conversations = self.get_all_client_conversations(client_id)
            
            if not all_conversations:
                print(f"No conversations found for client {client_id}")
                return "No conversations found for this client."
            
            print(f"Found {len(all_conversations)} conversations to summarize")
            
            # Prepare conversation history for the prompt
            conversation_history = []
            total_messages = 0
            
            for conv in all_conversations:
                messages = conv.get("conversation", [])
                total_messages += len(messages)
                conv_summary = {
                    "start_time": conv.get("start_time"),
                    "end_time": conv.get("end_time", "ongoing"),
                    "status": conv.get("status"),
                    "messages": messages
                }
                conversation_history.append(conv_summary)
            
            print(f"Total messages across all conversations: {total_messages}")

            if total_messages == 0:
                print("No messages found in conversations")
                return "No messages found in conversations for this client."

            # Add metadata to the prompt
            prompt = f"""
            Generate a comprehensive summary for Client ID: {client_id} based on all their conversations.
            Format the response in clear sections with minimal markdown.
            
            Total Conversations: {len(all_conversations)}
            Total Messages: {total_messages}
            Time Span: {conversation_history[0]["start_time"]} to {conversation_history[-1]["end_time"]}
            
            Focus on (but do not state if repetitive content is present):
            1. Client's preferences and requirements:
               - Budget range
               - Preferred locations
               - Property size requirements
               - Must-have amenities
            2. Evolution of requirements over time
            3. Key discussion points from all conversations
            4. Decisions made or changed during discussions
            5. Current status and next steps
            
            Conversation History:
            {json.dumps(conversation_history, indent=2)}
            
            Please provide a structured summary that shows the progression of the client's requirements and decisions over time.
            Use simple formatting with clear section headers and bullet points.
            """
            
            print("Generating summary using Gemini...")
            response = self.model.generate_content(prompt)
            
            if not response.text:
                raise ValueError("No response generated from the model")
            
            # Format the response
            formatted_response = self.format_summary_response(response.text)
                
            # Store summary in client's directory
            client_dir = CONVERSATIONS_DIR / client_id
            client_dir.mkdir(exist_ok=True)
            
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": formatted_response,
                "total_conversations": len(all_conversations),
                "total_messages": total_messages,
                "conversations_included": [
                    {
                        "start_time": conv.get("start_time"),
                        "end_time": conv.get("end_time", "ongoing"),
                        "status": conv.get("status"),
                        "message_count": len(conv.get("conversation", []))
                    }
                    for conv in all_conversations
                ]
            }
            
            # Save as JSON for better structure and metadata
            summary_file = client_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print("Summary generated and stored successfully")
            return formatted_response
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def get_property_recommendations(self, client_id):
        """Get property recommendations based on all conversations"""
        try:
            # First check if we have any properties
            if self.properties_df.empty:
                return {
                    'status': 'error',
                    'message': 'No properties available in the system'
                }

            # Get available properties
            available_properties = self.properties_df[
                self.properties_df['status'] == 'available'
            ].to_dict('records')
            
            if not available_properties:
                return {
                    'status': 'error',
                    'message': 'No available properties found'
                }

            # Get the client summary
            summary = self.get_conversation_summary(client_id)
            
            # If no conversations exist yet, provide default recommendations
            if summary == "No conversations found for this client.":
                # Return top 3 available properties as default recommendations
                default_properties = available_properties[:3]
                recommendations = "=== DEFAULT RECOMMENDATIONS ===\n"
                recommendations += "Since there are no conversations yet, here are our top available properties:\n\n"
                
                for prop in default_properties:
                    recommendations += f"=== {prop['type']} in {prop['location']} ===\n"
                    recommendations += f"  • Property ID: {prop['id']}\n"
                    recommendations += f"  • Price: ₹{prop['price']:,}\n"
                    recommendations += f"  • Size: {prop['size']} sq ft\n"
                    recommendations += f"  • Type: {prop['type']}\n"
                    if prop.get('amenities'):
                        amenities_list = prop['amenities'].split(',') if isinstance(prop['amenities'], str) else prop['amenities']
                        recommendations += f"  • Amenities: {', '.join(amenities_list)}\n"
                    recommendations += "\n"
                
                return {
                    'status': 'success',
                    'data': {
                        'recommendations': recommendations
                    }
                }

            # Generate recommendations using Gemini
            prompt = f"""
            Based on this client summary: {summary}
            
            And these available properties:
            {json.dumps(available_properties, indent=2)}
            
            Recommend the top 3 most suitable properties that match the client's requirements. Consider:
            1. Budget alignment with client's stated range
            2. Location preferences mentioned
            3. Size requirements
            4. Property type preferences
            5. Amenities that match client's requirements
            6. Any specific requirements or preferences mentioned
            
            Format each recommendation as:
            === PROPERTY TYPE in LOCATION ===
            • Property ID: [ID]
            • Price: [Price in ₹]
            • Size: [Size in sq ft]
            • Type: [Property Type]
            • Amenities: [List all amenities]
            • Key Benefits: [List key benefits based on client requirements, highlighting matching amenities]
            
            If no suitable properties are found, clearly state that no matches were found.
            """
            
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return {
                    'status': 'error',
                    'message': 'No recommendations could be generated'
                }
            
            # Format the response
            formatted_response = self.format_summary_response(response.text)
            
            return {
                'status': 'success',
                'data': {
                    'recommendations': formatted_response
                }
            }
            
        except Exception as e:
            print(f"Error in get_property_recommendations: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_property_locations(self):
        """Get all property locations with their details"""
        properties = []
        for _, row in self.properties_df.iterrows():
            property_data = {
                'id': row['id'],
                'location': row['location'],
                'price': row['price'],
                'size': row['size'],
                'amenities': row['amenities'].split(',') if isinstance(row['amenities'], str) else [],
                'position': {
                    'lat': float(row['latitude']),
                    'lng': float(row['longitude'])
                }
            }
            properties.append(property_data)
        return properties

    def add_user(self, name, email, phone):
        """Add a new user to the system"""
        try:
            # Generate a unique user ID
            user_id = f"USER_{len(self.users_df) + 1}"
            
            # Create new user entry
            new_user = {
                'user_id': user_id,
                'name': name,
                'email': email,
                'phone': phone,
                'created_at': datetime.now().isoformat()
            }
            
            # Add to DataFrame
            self.users_df = pd.concat([self.users_df, pd.DataFrame([new_user])], ignore_index=True)
            
            # Save to CSV
            self.users_df.to_csv('users.csv', index=False)
            print(f"Successfully created user: {user_id}")
            
            return new_user
        except Exception as e:
            print(f"Error adding user: {e}")
            raise

assistant = RealEstateAssistant()

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        data = request.json
        audio_data = np.array(data['audio'], dtype=np.float32)
        sample_rate = int(data['sampleRate'])
        language = data['language']
        
        text = assistant.transcribe_audio(audio_data, sample_rate, language)
        if text:
            # Translate to English if not already in English
            if language != 'en':
                translated = assistant.translate_text(text, 'en')
                return jsonify({
                    'original': text,
                    'translated': translated
                })
            return jsonify({
                'original': text,
                'translated': text
            })
        return jsonify({'error': 'Failed to transcribe audio'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data['text']
        language = data['language']
        
        audio_buffer = assistant.text_to_speech(text, language)
        if audio_buffer:
            return send_file(
                audio_buffer,
                mimetype='audio/mp3',
                as_attachment=True,
                download_name='speech.mp3'
            )
        return jsonify({'error': 'Failed to generate speech'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_speech():
    data = request.json
    text = data.get('text')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'en')
    
    translated = assistant.translate_text(text, target_lang)
    return jsonify({'translated': translated})

@app.route('/api/store_conversation', methods=['POST'])
def store_conversation():
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
            
        client_id = data.get('client_id')
        message = data.get('message')
        
        # Log received data for debugging
        print(f"Received store_conversation request - client_id: {client_id}, message: {message}")
        
        if not client_id:
            return jsonify({
                'status': 'error',
                'message': 'client_id is required'
            }), 400
            
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'message is required'
            }), 400
            
        # Validate message structure
        if not isinstance(message, dict):
            return jsonify({
                'status': 'error',
                'message': 'message must be an object'
            }), 400
            
        required_fields = ['text', 'translated', 'language']
        missing_fields = [field for field in required_fields if field not in message]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'message is missing required fields: {", ".join(missing_fields)}'
            }), 400
            
        try:
            assistant.store_conversation(client_id, message)
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Error in store_conversation: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    except Exception as e:
        print(f"Error processing store_conversation request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid request format'
        }), 400

@app.route('/api/end_conversation', methods=['POST'])
def end_conversation():
    data = request.json
    client_id = data.get('client_id')
    
    success = assistant.end_conversation(client_id)
    return jsonify({'status': 'success' if success else 'no active conversation'})

@app.route('/api/get_summary/<client_id>', methods=['GET'])
def get_summary(client_id):
    summary = assistant.get_conversation_summary(client_id)
    return jsonify({'summary': summary})

@app.route('/api/mark_done', methods=['POST'])
def mark_done():
    try:
        data = request.json
        client_id = data.get('client_id')
        property_id = data.get('property_id')
        close_type = data.get('close_type', 'purchase')
        
        if not client_id:
            return jsonify({
                'status': 'error',
                'message': 'client_id is required'
            }), 400
            
        if close_type == 'purchase' and not property_id:
            return jsonify({
                'status': 'error',
                'message': 'property_id is required for purchase type'
            }), 400

        success = assistant.mark_client_done(client_id, property_id if close_type == 'purchase' else None)
        if success:
            return jsonify({'status': 'success'})
        return jsonify({
            'status': 'error',
            'message': 'Failed to mark client as done'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get_recommendations/<client_id>', methods=['GET'])
def get_recommendations(client_id):
    try:
        if not client_id:
            return jsonify({
                'status': 'error',
                'message': 'Client ID is required'
            }), 400

        # Check if client exists
        client_exists = any(assistant.clients_df['client_id'] == client_id)
        if not client_exists:
            return jsonify({
                'status': 'error',
                'message': 'Client not found'
            }), 404

        # Get recommendations from the assistant
        recommendations = assistant.get_property_recommendations(client_id)
        
        # Return the recommendations directly - it already has the correct structure
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/property-locations', methods=['GET'])
def get_property_locations():
    try:
        # Get properties from the assistant's properties DataFrame
        properties_list = []
        for _, row in assistant.properties_df.iterrows():
            property_data = {
                'id': str(row['id']),
                'location': row['location'],
                'price': float(row['price']),
                'size': float(row['size']),
                'amenities': row['amenities'].split(',') if isinstance(row['amenities'], str) else [],
                'city': row['location'].split(',')[0].strip(),
                'position': {
                    'lat': float(row['latitude']),
                    'lng': float(row['longitude'])
                }
            }
            properties_list.append(property_data)
        
        return jsonify({
            'status': 'success',
            'properties': properties_list
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/extract_meeting_details', methods=['POST'])
def extract_meeting_details():
    try:
        data = request.json
        client_id = data.get('client_id')
        
        # Get all conversations for the client
        all_conversations = assistant.get_all_client_conversations(client_id)
        if not all_conversations:
            return jsonify({'error': 'No conversations found'}), 404
        
        # Get the most recent conversation
        latest_conversation = all_conversations[-1]
        
        # Prepare the conversation text for Gemini
        messages = latest_conversation.get('conversation', [])
        conversation_text = "\n".join([
            f"[{msg.get('timestamp', '')}] {msg.get('translated_text', '')}"
            for msg in messages
        ])
        
        # Use the same model instance from RealEstateAssistant
        prompt = f"""
        Your task is to extract meeting details from the conversation below. 
        Look for any mentions of scheduling a meeting, viewing a property, or future appointments.
        
        Rules:
        1. Return ONLY a valid JSON object with the following fields:
           - date: the date of the meeting in YYYY-MM-DD format
           - time: the time of the meeting in HH:MM format (24-hour)
           - location: the location of the meeting
           - details: any additional details about the meeting
        2. If no meeting details are found, return exactly: null
        3. If multiple meetings are mentioned, use the most recently discussed one
        4. Convert relative dates (like "tomorrow", "next week") to absolute dates based on the message timestamps
        5. If only date is mentioned without time, assume 10:00 AM (10:00)
        6. If only time is mentioned without date, assume the next occurrence of that time
        7. If neither date nor time is mentioned but a meeting is clearly planned, assume next business day at 10:00 AM
        
        Example valid responses:
        {{"date": "2024-03-10", "time": "14:30", "location": "123 Park Avenue", "details": "Property viewing with agent John"}}
        {{"date": "2024-03-11", "time": "10:00", "location": "456 Oak Street", "details": "Initial consultation"}}
        null
        
        Conversation:
        {conversation_text}
        
        Return ONLY the JSON response without any additional text or explanation:
        """
        
        response = assistant.model.generate_content(prompt)
        
        try:
            # Clean the response text to ensure it's valid JSON
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
            if cleaned_response.lower() == 'null':
                return jsonify({'meetingDetails': None})
            
            meeting_details = json.loads(cleaned_response)
            
            # If time is not provided, set default time
            if 'time' not in meeting_details or not meeting_details['time']:
                meeting_details['time'] = '10:00'
            
            # If date is not provided, set to next business day
            if 'date' not in meeting_details or not meeting_details['date']:
                next_day = datetime.now() + timedelta(days=1)
                while next_day.weekday() >= 5:  # Skip weekends
                    next_day += timedelta(days=1)
                meeting_details['date'] = next_day.strftime('%Y-%m-%d')
            
            return jsonify({'meetingDetails': meeting_details})
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response.text}")
            return jsonify({'meetingDetails': None})
            
    except Exception as e:
        print(f"Error in extract_meeting_details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_calendar_event', methods=['POST'])
def create_calendar_event():
    try:
        data = request.json
        meeting_details = data.get('meeting_details')
        client_id = data.get('client_id')
        
        if not meeting_details:
            return jsonify({'error': 'Meeting details are required'}), 400
        
        # Get Google Calendar credentials
        creds = get_calendar_credentials()
        service = build('calendar', 'v3', credentials=creds)
        
        # Create the event
        start_datetime = f"{meeting_details['date']}T{meeting_details['time']}:00"
        # Set duration to 1 hour by default
        end_datetime = datetime.fromisoformat(start_datetime).replace(hour=datetime.fromisoformat(start_datetime).hour + 1).isoformat()
        
        event = {
            'summary': f'Meeting with {client_id}',  # Standardized event name
            'location': meeting_details['location'],
            'description': meeting_details['details'],
            'start': {
                'dateTime': start_datetime,
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': 'Asia/Kolkata',
            },
        }
        
        event = service.events().insert(calendarId='primary', body=event).execute()
        return jsonify({'eventId': event['id']})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<user_id>/clients', methods=['GET'])
def get_user_clients(user_id):
    """Get all clients for a specific user"""
    try:
        # First check if user exists
        user = assistant.get_user_details(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
            
        try:
            clients = assistant.get_user_clients(user_id)
            print(f"Returning clients for user {user_id}:", clients)  # Debug log
            return jsonify({
                'status': 'success',
                'clients': clients if clients is not None else []
            })
        except ValueError as ve:
            return jsonify({
                'status': 'error',
                'message': str(ve)
            }), 400
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error retrieving clients: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error in get_user_clients endpoint: {str(e)}")  # Log the error
        return jsonify({
            'status': 'error',
            'message': str(e) if str(e) else 'An unknown error occurred'
        }), 500

@app.route('/api/users/<user_id>/clients', methods=['POST'])
def create_user_client(user_id):
    """Create a new client for a specific user"""
    try:
        # First check if user exists
        user = assistant.get_user_details(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404

        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        client_name = data.get('client_name')
        if not client_name:
            return jsonify({
                'status': 'error',
                'message': 'Client name is required'
            }), 400

        try:
            client = assistant.create_client(user_id, client_name)
            return jsonify({
                'status': 'success',
                'client': client
            })
        except Exception as e:
            print(f"Error creating client: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to create client: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error in create_user_client endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    try:
        users = assistant.get_all_users()
        return jsonify({
            'status': 'success',
            'users': users
        })
    except Exception as e:
        print(f"Error getting users: {str(e)}")  # Log the error
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users', methods=['POST'])
def add_user():
    """Add a new user"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        
        if not all([name, email, phone]):
            return jsonify({
                'status': 'error',
                'message': 'Name, email, and phone are required'
            }), 400
        
        # Check if user with same email already exists
        existing_user = assistant.users_df[assistant.users_df['email'] == email]
        if not existing_user.empty:
            return jsonify({
                'status': 'error',
                'message': 'User with this email already exists'
            }), 400
        
        try:
            user = assistant.add_user(name, email, phone)
            return jsonify({
                'status': 'success',
                'user': user
            })
        except Exception as e:
            print(f"Error in add_user: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to create user: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error in add_user endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get_available_properties', methods=['GET'])
def get_available_properties():
    try:
        # Get all properties from the database
        properties_dir = os.path.join(BASE_DIR, 'properties')
        properties_file = os.path.join(properties_dir, 'properties.json')
        
        if not os.path.exists(properties_file):
            return jsonify({
                'status': 'success',
                'properties': [],
                'purchased_properties': []
            })
        
        with open(properties_file, 'r') as f:
            all_properties = json.load(f)
        
        # Get list of purchased properties
        purchased_properties = set()
        clients_dir = os.path.join(BASE_DIR, 'clients')
        if os.path.exists(clients_dir):
            for client_id in os.listdir(clients_dir):
                client_file = os.path.join(clients_dir, client_id, 'client_info.json')
                if os.path.exists(client_file):
                    with open(client_file, 'r') as f:
                        client_data = json.load(f)
                        if client_data.get('purchase_property'):
                            purchased_properties.add(client_data['purchase_property'])
        
        return jsonify({
            'status': 'success',
            'properties': all_properties,
            'purchased_properties': list(purchased_properties)
        })
        
    except Exception as e:
        print(f"Error getting available properties: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get available properties: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 