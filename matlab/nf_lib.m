function L = nf_lib()
%NF_LIB  Shared model for TTD-free wideband near-field DFT beam training.
%
%   L = nf_lib() returns a struct of function handles. See run_theory_checks.m
%   and run_threshold_mc.m for usage.
%
%   Conventions (must match analysis/THEORY.md):
%     ULA on the y-axis, N elements, spacing d = lambda_c/2 fixed by the CENTRE
%     frequency (phase shifters, no true time delay). Spatial angle
%     theta = sin(AoD) in [-1,1]. Codewords are frequency-independent far-field
%     steering vectors designed at f_c and reused on every subcarrier.
%
%     eta_m = f_m/f_c,  beta = B/f_c
%     Zooming map      theta_m = eta_m*theta
%                      r_m     = r*(1-eta_m^2*theta^2)/(eta_m*(1-theta^2))
%     Pattern width    W     = N^2*d*(1-theta^2)/(2r)          [DFT beams]
%     Dither span      Delta = beta*N*abs(theta)/2             [DFT beams]
%     Sampling gap     g     = max(dec-Delta, Delta/(M-1))
%     Effective count  E_eff = sum(G.^2)/max(G(:).^2) ~ W/g
%
%   Amplitude-only throughout: carrier phase is never used.

L.C0        = 3e8;
L.array     = @array;
L.subc      = @subc;
L.gain      = @gain;
L.pattern   = @pattern;
L.width     = @width_beams;
L.dither    = @dither_beams;
L.gap       = @gap_beams;
L.eeff      = @eeff;
L.dict      = @dict;
L.estimate  = @estimate;
end

% ---------------------------------------------------------------- model

function a = array(N, fc)
a.N = N; a.fc = fc; a.c = 3e8;
a.d = a.c/(2*fc);                                   % half wavelength at f_c
a.dn = ((2*(1:N) - N - 1)/2).';                     % element index offsets
a.grid = (2*(1:N) - N - 1)/N;                       % DFT codebook grid
end

function fm = subc(a, beta, M)
if M == 1, fm = a.fc; return; end
fm = a.fc + beta*a.fc/M * ((1:M) - 1 - (M-1)/2);
end

function G = gain(a, theta, r, fm, phis)
%GAIN  |b_m^H(theta,r) a(phi)| for each subcarrier (rows) and codeword (cols).
fm = fm(:); phis = phis(:);
rn = sqrt(r^2 + (a.dn*a.d).^2 - 2*r*theta*a.dn*a.d);   % exact spherical wave
ph = exp(1i*2*pi*(fm*(rn - r).')/a.c);                 % M x N
A  = exp(-1i*pi*(phis*a.dn.'));                        % K x N
G  = abs(ph*A')/a.N;                                   % M x K
end

function [Wb, Cb] = pattern(a, theta, r, fm, phis, thr)
%PATTERN  measured half-max support width and centre, in DFT-beam units.
if nargin < 6, thr = 0.5; end
G = gain(a, theta, r, fm, phis);
Wb = zeros(size(G,1),1); Cb = zeros(size(G,1),1);
for m = 1:size(G,1)
    s = phis(G(m,:) > thr*max(G(m,:)));
    Wb(m) = (max(s) - min(s))/(2/a.N);
    Cb(m) = mean(s)/(2/a.N);
end
end

% ------------------------------------------------------------- analytic

function W = width_beams(a, theta, r)
W = a.N^2 * a.d * (1 - theta^2) / (2*r);
end

function D = dither_beams(a, beta, theta)
D = beta * a.N * abs(theta) / 2;
end

function g = gap_beams(a, beta, theta, dec, M)
if M == 1, g = dec; return; end
D = dither_beams(a, beta, theta);
g = max(dec - D, D/(M-1));
end

function E = eeff(a, theta, r, fm, phis)
G = gain(a, theta, r, fm, phis);
E = sum(G(:).^2) / max(G(:).^2);
end

% ------------------------------------------------------------ estimator

function D = dict(a, fm, phis, thGrid, uGrid)
%DICT  |G| on the (theta, 1/r) grid. Independent of data, so build once.
D = zeros(numel(thGrid), numel(uGrid), numel(fm), numel(phis));
for i = 1:numel(thGrid)
    for j = 1:numel(uGrid)
        D(i,j,:,:) = gain(a, thGrid(i), 1/uGrid(j), fm, phis);
    end
end
end

function [thHat, rHat] = estimate(a, z, fm, phis, thGrid, uGrid, D, rMin, rMax)
%ESTIMATE  profiled amplitude least squares. The per-subcarrier gain A_m is
%eliminated in closed form, so the criterion ignores frequency-dependent path
%loss and never touches carrier phase:
%     maximise  sum_m <z_m,G_m>^2 / ||G_m||^2
num = squeeze(sum(sum(bsxfun(@times, D, reshape(z, [1 1 size(z)])), 4).^2, 3));
den = squeeze(sum(sum(D.^2, 4), 3));
S = num ./ max(den, 1e-30);
[~, k] = max(S(:));
[i, j] = ind2sub(size(S), k);

obj = @(p) -profiled_ls(z, gain(a, min(max(p(1),-0.95),0.95), ...
                                  min(max(p(2),rMin),rMax), fm, phis));
p = fminsearch(obj, [thGrid(i); 1/uGrid(j)], ...
               optimset('TolX',1e-5,'TolFun',1e-8,'MaxIter',300,'Display','off'));
thHat = min(max(p(1),-0.95),0.95);
rHat  = min(max(p(2),rMin),rMax);
end

function v = profiled_ls(z, G)
v = sum( sum(z.*G,2).^2 ./ max(sum(G.^2,2),1e-30) );
end
