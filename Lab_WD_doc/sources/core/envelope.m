function [ env ] = envelope( E )
%ENVELOPE Envelope of oscillating signal
% E is a real vector.
% If E is two-dimensional the operation is carried out columnwise.
%
% This function works by finding out the peak of the Fourier transform of E
% and performing a moving-window average of the function. The window is
% approximately top hat.

N = length(E);

NFFT = 2^nextpow2(N);
% If E has more than 1 dimension then the only the central column is
% transformed (works well for a signal in the middle of the window)
trans = fft(E(:,round(size(E,2)/2)), NFFT);

% Get resonant wavelength (reslam)
[maxE, res] = max(abs(trans));
reslam = N/res;

% Generate time-averaging window
window = tophat(reslam);

% Apply time-averaging window
env = sqrt(2) * convn(abs(E),window,'same');

end

function W = tophat(L)
% Generates a top hat window -- when L is non-integer the last value of the
% window function is the remainder

span = floor(L);
window = ones(span+1,1);
window(end) = L-span;
W = window / sum(window);

end