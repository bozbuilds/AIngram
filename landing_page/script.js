document.addEventListener('DOMContentLoaded', () => {
    // Copy button
    const copyBtn = document.getElementById('copyBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText('pip install aingram').then(() => {
                const icon = copyBtn.innerHTML;
                copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                setTimeout(() => {
                    copyBtn.innerHTML = icon;
                }, 2000);
            });
        });
    }

    // Form submission
    const form = document.getElementById('waitlistForm');
    const messageEl = document.getElementById('formMessage');
    const submitBtn = document.getElementById('submitBtn');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('emailInput').value;
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Joining...';
            messageEl.textContent = '';
            messageEl.className = 'form-message';

            try {
                const response = await fetch('/api/waitlist', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email })
                });

                const data = await response.json();

                if (response.ok) {
                    messageEl.textContent = 'Thanks for joining the waitlist! We will be in touch soon.';
                    messageEl.className = 'form-message success';
                    form.reset();
                } else {
                    messageEl.textContent = data.error || 'Something went wrong. Please try again.';
                    messageEl.className = 'form-message error';
                }
            } catch (error) {
                messageEl.textContent = 'Network error. Please try again later.';
                messageEl.className = 'form-message error';
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Join Waitlist';
            }
        });
    }
});
