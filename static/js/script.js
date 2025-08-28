document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const chatHistory = document.getElementById('chat-history');
    const userMessageText = chatInput.value.trim();

    if (userMessageText === "") return;

    // Display user message
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user-message';
    userMessageDiv.textContent = userMessageText;
    chatHistory.appendChild(userMessageDiv);

    // Clear input box
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Call the backend API
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: userMessageText }),
        });

        const data = await response.json();
        const botResponseText = data.response;

        // Display bot message
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'message bot-message';
        botMessageDiv.textContent = botResponseText;
        chatHistory.appendChild(botMessageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

    } catch (error) {
        console.error('Error:', error);
        const errorMessageDiv = document.createElement('div');
        errorMessageDiv.className = 'message bot-message';
        errorMessageDiv.textContent = 'Sorry, there was an error processing your request.';
        chatHistory.appendChild(errorMessageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}
