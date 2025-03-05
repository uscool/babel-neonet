import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
  Container,
  Paper,
  Typography,
  Button,
  TextField,
  Box,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
} from '@mui/material';
import PropertyMap from './components/PropertyMap';
import UserManagement from './components/UserManagement';
import PropertyTable from './components/PropertyTable';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedClient, setSelectedClient] = useState(null);
  const [conversation, setConversation] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('hi-IN');
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [summary, setSummary] = useState('');
  const [recommendations, setRecommendations] = useState('');
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedProperty, setSelectedProperty] = useState('');
  const [error, setError] = useState('');
  const [isConversationActive, setIsConversationActive] = useState(false);
  const [meetingDetails, setMeetingDetails] = useState(null);
  const [calendarDialogOpen, setCalendarDialogOpen] = useState(false);
  const [markDoneType, setMarkDoneType] = useState('purchase');
  const [properties, setProperties] = useState([]);
  
  const recognition = useRef(null);

  const languages = useMemo(() => [
    { code: 'hi-IN', name: 'Hindi' },
    { code: 'ta-IN', name: 'Tamil' },
    { code: 'te-IN', name: 'Telugu' },
    { code: 'kn-IN', name: 'Kannada' },
    { code: 'ml-IN', name: 'Malayalam' },
    { code: 'mr-IN', name: 'Marathi' },
  ], []);

  const handleTranslate = useCallback(async (text) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/translate`, {
        text,
        source_lang: selectedLanguage.split('-')[0],
        target_lang: targetLanguage.split('-')[0]
      });
      return response.data.translated;
    } catch (error) {
      setError('Translation error: ' + error.message);
      return text;
    }
  }, [selectedLanguage, targetLanguage]);

  const handleSendMessage = useCallback(async (originalText, translatedText) => {
    console.log('Attempting to send message with client:', selectedClient);
    if (!selectedClient?.client_id) {
      console.error('No client selected or client_id missing:', selectedClient);
      setError('No client selected. Please select a client first.');
      return;
    }

    if (!isConversationActive) {
      console.error('Conversation not started');
      setError('Please start a new conversation first.');
      return;
    }

    if (!originalText) {
      setError('No message text provided.');
      return;
    }

    try {
      const translation = translatedText || await handleTranslate(originalText);
      
      const message = { 
        text: originalText, 
        translated: translation,
        language: selectedLanguage
      };
      
      console.log('Sending message:', {
        client_id: selectedClient.client_id,
        message: message
      });

      const response = await axios.post(`${API_BASE_URL}/store_conversation`, {
        client_id: selectedClient.client_id,
        message: message
      });

      if (response.data.status === 'success') {
        setConversation(prev => [...prev, message]);
        setError('');
      } else {
        throw new Error(response.data.message || 'Failed to store conversation');
      }
    } catch (error) {
      console.error('Error in handleSendMessage:', error);
      const errorMessage = error.response?.data?.message || error.message;
      setError(`Error storing conversation: ${errorMessage}`);
      
      console.error('Full error:', error);
      if (error.response) {
        console.error('Response data:', error.response.data);
        console.error('Response status:', error.response.status);
      }
      
      setConversation(prev => [...prev, { 
        text: originalText,
        translated: translatedText || originalText,
        language: selectedLanguage,
        error: true 
      }]);
    }
  }, [selectedClient, isConversationActive, selectedLanguage, handleTranslate]);

  const startRecording = async () => {
    try {
      if (!recognition.current) {
        setError('Speech recognition not initialized');
        return;
      }

      if (!selectedClient?.client_id || !isConversationActive) {
        setError('Please start a conversation first');
        return;
      }

      console.log('Starting recording...');
      recognition.current.continuous = true;
      recognition.current.interimResults = false;
      recognition.current.lang = selectedLanguage;
      
      await recognition.current.start();
      setIsRecording(true);
    } catch (error) {
      if (error.name === 'NotAllowedError') {
        setError('Please allow microphone access to record');
      } else {
        console.error('Error in startRecording:', error);
        setError('Error starting recording: ' + error.message);
      }
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    try {
      console.log('Stopping recording...');
      if (recognition.current) {
        recognition.current.stop();
      }
      setIsRecording(false);
    } catch (error) {
      console.error('Error in stopRecording:', error);
      setError('Error stopping recording: ' + error.message);
    }
  };

  useEffect(() => {
    let recognitionInstance = null;

    if ('webkitSpeechRecognition' in window) {
      recognitionInstance = new window.webkitSpeechRecognition();
      recognition.current = recognitionInstance;

      recognitionInstance.onstart = () => {
        console.log('Speech recognition started');
        setIsRecording(true);
      };

      recognitionInstance.onend = () => {
        console.log('Speech recognition ended');
        setIsRecording(false);
      };

      recognitionInstance.onresult = async (event) => {
        const text = event.results[event.results.length - 1][0].transcript;
        console.log('Recognized text:', text);
        try {
          await handleSendMessage(text);
        } catch (error) {
          setError('Error processing speech: ' + error.message);
        }
      };

      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setError('Speech recognition error: ' + event.error);
        setIsRecording(false);
      };
    } else {
      setError('Speech recognition is not supported in this browser');
    }

    return () => {
      if (recognitionInstance) {
        try {
          recognitionInstance.stop();
        } catch (error) {
          console.error('Error stopping recognition on cleanup:', error);
        }
      }
    };
  }, [handleSendMessage]);

  useEffect(() => {
    if (recognition.current && selectedLanguage) {
      recognition.current.lang = selectedLanguage;
    }
  }, [selectedLanguage]);

  const handleUserSelect = (user) => {
    setSelectedUser(user);
    // Reset client-related state when user changes
    setSelectedClient(null);
    setConversation([]);
    setIsConversationActive(false);
    setSummary('');
    setRecommendations('');
  };

  const handleClientSelect = (client) => {
    console.log('Selecting client:', client);
    setSelectedClient(client);
    // Reset conversation state when client changes
    setConversation([]);
    setIsConversationActive(false);
    setSummary('');
    setRecommendations('');
  };

  const handleStartConversation = () => {
    console.log('Starting conversation with client:', selectedClient);
    console.log('Current isConversationActive state:', isConversationActive);
    
    if (!selectedClient) {
      console.log('No client selected, showing error');
      setError('Please select a client first');
      return;
    }
    
    console.log('Setting conversation active to true');
    setIsConversationActive(true);
    setConversation([]);
    
    // Add a timeout to verify the state was updated
    setTimeout(() => {
      console.log('Conversation active state after update:', isConversationActive);
    }, 100);
  };

  const handleEndConversation = async () => {
    if (!selectedUser) {
      setError('Please select a user first');
      return;
    }

    if (!selectedClient) {
      setError('Please select a client first');
      return;
    }

    try {
      // Stop recording if it's active
      if (isRecording) {
        stopRecording();
      }

      // End the conversation
      await axios.post(`${API_BASE_URL}/end_conversation`, {
        client_id: selectedClient.client_id
      });

      // Extract meeting details
      const meetingResponse = await axios.post(`${API_BASE_URL}/extract_meeting_details`, {
        client_id: selectedClient.client_id
      });

      if (meetingResponse.data.meetingDetails) {
        setMeetingDetails(meetingResponse.data.meetingDetails);
        setCalendarDialogOpen(true);
      } else {
        setError('No meeting details found in the conversation. Make sure to discuss meeting date, time, and location.');
      }

      // Reset conversation state regardless of meeting details
      setIsConversationActive(false);
      setConversation([]);

    } catch (error) {
      console.error('Error ending conversation:', error);
      const errorMessage = error.response?.data?.error || error.message;
      setError(`Error ending conversation: ${errorMessage}`);
      
      // Still reset conversation state even if there's an error
      setIsConversationActive(false);
      setConversation([]);
    }
  };

  const handleCreateCalendarEvent = async () => {
    if (!selectedUser || !meetingDetails) return;

    try {
      await axios.post(`${API_BASE_URL}/create_calendar_event`, {
        meeting_details: meetingDetails,
        client_id: selectedClient?.client_id,
        user_id: selectedUser.user_id
      });
      setCalendarDialogOpen(false);
    } catch (error) {
      console.error('Error creating calendar event:', error);
    }
  };

  const handleGetSummary = async (clientId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/get_summary/${clientId}`);
      setSummary(response.data.summary);
    } catch (error) {
      setError('Error getting summary: ' + error.message);
    }
  };

  const handleGetRecommendations = async (clientId) => {
    try {
      if (!clientId) {
        setError('Please select a client first');
        return;
      }

      // Clear previous recommendations and errors
      setRecommendations('');
      setError('');
      
      console.log('Fetching recommendations for client:', clientId);
      const response = await axios.get(`${API_BASE_URL}/get_recommendations/${clientId}`);
      console.log('Recommendations response:', response.data);

      // Handle the response based on status
      if (response.data.status === 'success' && response.data.data?.recommendations) {
        // Set recommendations if available
        setRecommendations(response.data.data.recommendations);
      } else if (response.data.status === 'error') {
        // Handle error case
        setError(response.data.message || 'Failed to get recommendations');
      } else {
        // Handle unexpected response format
        setError('Unexpected response format from server');
      }
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setError(error.response?.data?.message || error.message || 'Error fetching recommendations');
      setRecommendations('');
    }
  };

  const handleMarkDone = async () => {
    try {
      if (markDoneType === 'purchase' && !selectedProperty) {
        setError('Please enter a property ID');
        return;
      }

      const response = await axios.post(`${API_BASE_URL}/mark_done`, {
        client_id: selectedClient?.client_id,
        property_id: markDoneType === 'purchase' ? selectedProperty : null,
        close_type: markDoneType
      });

      if (response.data.status === 'success') {
        setOpenDialog(false);
        // Refresh client status
        if (selectedClient) {
          const updatedClient = { 
            ...selectedClient, 
            status: 'closed',
            purchase_property: markDoneType === 'purchase' ? selectedProperty : null,
            purchase_date: new Date().toISOString().split('T')[0]
          };
          handleClientSelect(updatedClient);
          
          // Update properties list to remove purchased property
          if (markDoneType === 'purchase') {
            setProperties(prevProperties => 
              prevProperties.filter(prop => prop.id !== selectedProperty)
            );
          }
          
          // Clear recommendations if they exist since property status has changed
          setRecommendations('');
          
          // Show success message
          setError('');
        }
      } else {
        throw new Error(response.data.message || 'Failed to mark client as done');
      }
    } catch (error) {
      console.error('Error marking client as done:', error);
      setError('Error marking client as done: ' + (error.response?.data?.message || error.message));
    }
  };

  useEffect(() => {
    console.log('isConversationActive changed to:', isConversationActive);
  }, [isConversationActive]);

  useEffect(() => {
    const fetchProperties = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/get_available_properties`);
        if (response.data.status === 'success') {
          // Get all clients to check for purchased properties
          let purchasedProperties = new Set(response.data.purchased_properties || []);
          
          if (selectedUser) {
            try {
              const clientsResponse = await axios.get(`${API_BASE_URL}/users/${selectedUser.user_id}/clients`);
              if (clientsResponse.data.status === 'success') {
                clientsResponse.data.clients.forEach(client => {
                  if (client.purchase_property) {
                    purchasedProperties.add(client.purchase_property);
                  }
                });
              }
            } catch (error) {
              console.error('Error fetching clients:', error);
            }
          }
          
          // Filter out purchased properties
          const availableProperties = response.data.properties.filter(prop => 
            !purchasedProperties.has(prop.id)
          );
          console.log('Available properties:', availableProperties);
          setProperties(availableProperties || []);
        }
      } catch (error) {
        console.error('Error fetching properties:', error);
      }
    };

    fetchProperties();
  }, [selectedUser]);

  return (
    <Container maxWidth="lg" sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ 
        mt: 4, 
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        <Typography 
          variant="h4" 
          component="h1" 
          gutterBottom 
          align="center"
          sx={{
            fontWeight: 700,
            letterSpacing: '-0.5px',
            color: 'primary.main',
            mb: 4,
            '& span': {
              color: 'text.primary'
            }
          }}
        >
          NeoNet <span>Enterprise</span>
        </Typography>

        {/* User Management Section */}
        <Box sx={{ width: '100%', maxWidth: '800px' }}>
          <UserManagement 
            onUserSelect={handleUserSelect}
            onClientSelect={handleClientSelect}
          />
        </Box>

        {/* Selected User and Client Info */}
        {selectedUser && (
          <Paper sx={{ p: 2, mb: 2, width: '100%', maxWidth: '800px' }}>
            <Typography variant="subtitle1" align="center">
              Selected User: {selectedUser.name}
            </Typography>
            {selectedClient && (
              <Typography variant="subtitle1" align="center">
                Selected Client: {selectedClient.client_id} ({selectedClient.status})
              </Typography>
            )}
          </Paper>
        )}

        {/* Property Map and Table Section */}
        <Box sx={{ 
          display: 'flex', 
          gap: 2, 
          mt: 3,
          flexDirection: { xs: 'column', md: 'row' },
          height: { xs: 'auto', md: '500px' },
          width: '100%',
          justifyContent: 'center'
        }}>
          <Paper 
            elevation={3} 
            sx={{ 
              flex: { xs: '1 1 auto', md: '0 0 35%' }, 
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}
          >
            <Typography variant="h6" gutterBottom align="center">
              Property Map
            </Typography>
            <Box sx={{ flex: 1, minHeight: { xs: '300px', md: '400px' }, width: '100%' }}>
              <PropertyMap properties={properties} />
            </Box>
          </Paper>
          
          <Box sx={{ 
            flex: { xs: '1 1 auto', md: '0 0 63%' },
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
          }}>
            <PropertyTable properties={properties} />
          </Box>
        </Box>

        {/* Conversation Section */}
        <Paper sx={{ 
          p: 2, 
          mb: 2, 
          mt: 3,
          width: '100%',
          maxWidth: '1200px'
        }}>
          <Box sx={{ 
            mb: 2,
            display: 'flex',
            flexWrap: 'wrap',
            gap: 1,
            justifyContent: 'center'
          }}>
            {!isConversationActive ? (
              <Button
                variant="contained"
                color="primary"
                onClick={handleStartConversation}
                disabled={!selectedClient}
              >
                Start New Conversation
              </Button>
            ) : (
              <>
                <Button
                  variant="contained"
                  color={isRecording ? "error" : "primary"}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!selectedClient || !isConversationActive}
                >
                  {isRecording ? "Stop Recording" : "Start Recording"}
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={handleEndConversation}
                  disabled={isRecording}
                >
                  End Conversation
                </Button>
              </>
            )}
            <Button
              variant="outlined"
              onClick={() => handleGetSummary(selectedClient?.client_id)}
              disabled={!selectedClient}
            >
              Get Summary
            </Button>
            <Button
              variant="outlined"
              onClick={() => handleGetRecommendations(selectedClient?.client_id)}
              disabled={!selectedClient}
            >
              Get Recommendations
            </Button>
            <Button
              variant="outlined"
              onClick={() => setOpenDialog(true)}
              disabled={!selectedClient}
            >
              Mark as Done
            </Button>
          </Box>

          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center',
            alignItems: 'center',
            mb: 2,
            width: '100%',
            gap: 2
          }}>
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel>From</InputLabel>
              <Select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                label="From"
                sx={{ 
                  backgroundColor: 'white',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.dark',
                  }
                }}
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="hi-IN">Hindi</MenuItem>
                <MenuItem value="ta-IN">Tamil</MenuItem>
                <MenuItem value="te-IN">Telugu</MenuItem>
                <MenuItem value="kn-IN">Kannada</MenuItem>
                <MenuItem value="ml-IN">Malayalam</MenuItem>
                <MenuItem value="mr-IN">Marathi</MenuItem>
              </Select>
            </FormControl>

            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel>To</InputLabel>
              <Select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                label="To"
                sx={{ 
                  backgroundColor: 'white',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.dark',
                  }
                }}
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="hi-IN">Hindi</MenuItem>
                <MenuItem value="ta-IN">Tamil</MenuItem>
                <MenuItem value="te-IN">Telugu</MenuItem>
                <MenuItem value="kn-IN">Kannada</MenuItem>
                <MenuItem value="ml-IN">Malayalam</MenuItem>
                <MenuItem value="mr-IN">Marathi</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Box sx={{ 
            maxHeight: '300px', 
            overflow: 'auto',
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            p: 1,
            mx: 'auto'
          }}>
            <List>
              {conversation.map((msg, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={msg.text}
                    secondary={msg.translated}
                    primaryTypographyProps={{ align: 'center' }}
                    secondaryTypographyProps={{ align: 'center' }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </Paper>

        {/* Summary and Recommendations Section */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', md: 'row' },
          gap: 2,
          mt: 3,
          width: '100%',
          maxWidth: '1200px',
          justifyContent: 'center'
        }}>
          {summary && (
            <Paper sx={{ p: 2, flex: 1 }}>
              <Typography variant="h6" align="center">Conversation Summary</Typography>
              <Typography 
                component="pre"
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'inherit',
                  mt: 2,
                  textAlign: 'center',
                  '& .section': {
                    fontWeight: 'bold',
                    my: 1
                  },
                  '& .highlight': {
                    backgroundColor: '#f5f5f5',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontWeight: 'bold'
                  }
                }}
              >
                {summary}
              </Typography>
            </Paper>
          )}

          {recommendations && (
            <Paper sx={{ p: 2, flex: 1 }}>
              <Typography variant="h6" align="center">Property Recommendations</Typography>
              <Typography 
                component="pre"
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'inherit',
                  mt: 2,
                  textAlign: 'center',
                  '& .section': {
                    fontWeight: 'bold',
                    my: 1
                  },
                  '& .highlight': {
                    backgroundColor: '#f5f5f5',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontWeight: 'bold'
                  }
                }}
              >
                {recommendations}
              </Typography>
            </Paper>
          )}
        </Box>

        {/* Dialogs */}
        <Dialog 
          open={openDialog} 
          onClose={() => setOpenDialog(false)}
          PaperProps={{
            sx: { maxWidth: '400px', width: '100%', mx: 'auto' }
          }}
        >
          <DialogTitle align="center">Mark Client as Done</DialogTitle>
          <DialogContent>
            <FormControl fullWidth sx={{ mb: 2, mt: 1 }}>
              <InputLabel>Action Type</InputLabel>
              <Select
                value={markDoneType}
                onChange={(e) => setMarkDoneType(e.target.value)}
                label="Action Type"
              >
                <MenuItem value="purchase">Property Purchase</MenuItem>
                <MenuItem value="close">Close Client</MenuItem>
              </Select>
            </FormControl>
            
            {markDoneType === 'purchase' && (
              <TextField
                fullWidth
                label="Property ID"
                value={selectedProperty}
                onChange={(e) => setSelectedProperty(e.target.value)}
                sx={{ mt: 1 }}
              />
            )}
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }} align="center">
              {markDoneType === 'purchase' 
                ? 'Mark client as done with a property purchase. Please enter the property ID.'
                : 'Close the client relationship without a property purchase.'}
            </Typography>
          </DialogContent>
          <DialogActions sx={{ justifyContent: 'center', pb: 2 }}>
            <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
            <Button 
              onClick={handleMarkDone} 
              variant="contained"
              color={markDoneType === 'purchase' ? 'primary' : 'warning'}
            >
              {markDoneType === 'purchase' ? 'Confirm Purchase' : 'Close Client'}
            </Button>
          </DialogActions>
        </Dialog>

        <Dialog 
          open={calendarDialogOpen} 
          onClose={() => setCalendarDialogOpen(false)}
          PaperProps={{
            sx: { maxWidth: '400px', width: '100%', mx: 'auto' }
          }}
        >
          <DialogTitle align="center">Create Calendar Event</DialogTitle>
          <DialogContent>
            {meetingDetails && (
              <Box sx={{ textAlign: 'center' }}>
                <Typography>Date: {meetingDetails.date}</Typography>
                <Typography>Time: {meetingDetails.time}</Typography>
                <Typography>Location: {meetingDetails.location}</Typography>
                <Typography>Details: {meetingDetails.details}</Typography>
              </Box>
            )}
          </DialogContent>
          <DialogActions sx={{ justifyContent: 'center', pb: 2 }}>
            <Button onClick={() => setCalendarDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleCreateCalendarEvent} variant="contained" color="primary">
              Create Event
            </Button>
          </DialogActions>
        </Dialog>

        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={() => setError('')}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setError('')} severity="error">
            {error}
          </Alert>
        </Snackbar>
      </Box>
    </Container>
  );
}

export default App;
