%RUN_THEORY_CHECKS  Verify the analytic results of analysis/THEORY.md.
%
% Fast (seconds). Reproduces, in MATLAB, the propositions checked in Python:
%   1  zooming map        subcarrier m sees the centre-frequency pattern of a
%                         virtual user at (theta_m, r_m)
%   2  pattern features   centre and width both scale by the same eta_m, and
%                         W ~ N^2 d (1-theta^2)/(2r) in DFT-beam units
%   3  sampling rule      g = max(dec-Delta, Delta/(M-1)) and the pilot saving
%   4  E_eff ~ W/g        the quantity that sets the outlier threshold
%
% Expected (Python, N=256, f_c=40 GHz): map error <1.1%, W ratio 0.7-1.15,
% saving 1.9x at dec=8 and 4x at dec=4 for theta=0.6.

clear; clc;
L = nf_lib();
a = L.array(256, 40e9);
beta = 0.05; M = 9;

fprintf('--- 1. zooming map --------------------------------------------\n');
fprintf('%6s %6s %7s %8s %8s %10s %10s\n', ...
        'theta','r','eta','theta_m','r_m','rel err','vs naive');
for tr = [0.3 10; 0.6 10; 0.6 25; 0.8 15].'
    th = tr(1); r = tr(2);
    for eta = [1-beta/2, 1+beta/2]
        Gm    = L.gain(a, th, r, eta*a.fc, a.grid);
        th_m  = eta*th;
        r_m   = r*(1 - eta^2*th^2)/(eta*(1 - th^2));
        Gc    = L.gain(a, th_m, r_m, a.fc, a.grid);
        naive = L.gain(a, th,   r,   a.fc, a.grid);
        fprintf('%6.1f %6.0f %7.3f %8.4f %8.2f %9.2f%% %9.1f%%\n', th, r, eta, ...
                th_m, r_m, 100*max(abs(Gm-Gc))/max(Gm), 100*max(abs(Gm-naive))/max(Gm));
    end
end

fprintf('\n--- 2. pattern features ---------------------------------------\n');
fm = L.subc(a, beta, 5);
[Wb, Cb] = L.pattern(a, 0.5, 8, fm, a.grid);
eta = fm(:)/a.fc;
fprintf('centre/eta (should be constant): %s\n', mat2str(round(Cb./eta,3)));
fprintf('width /eta (should be constant): %s\n', mat2str(round(Wb./eta,3)));
fprintf('\n%6s %6s %12s %12s %7s\n','theta','r','W measured','W predicted','ratio');
for th = [0.2 0.4 0.6 0.8]
    for r = [8 15 30]
        Wm = L.pattern(a, th, r, a.fc, a.grid);
        Wp = L.width(a, th, r);
        fprintf('%6.1f %6.0f %12.1f %12.1f %7.2f\n', th, r, Wm, Wp, Wm/Wp);
    end
end

fprintf('\n--- 3. sampling rule and pilot saving -------------------------\n');
fprintf('%6s %5s %8s %8s %10s\n','theta','dec','Delta','g','saving');
for th = [0.2 0.3 0.6 0.8]
    for dec = [4 8 16]
        D = L.dither(a, beta, th);
        g = L.gap(a, beta, th, dec, M);
        fprintf('%6.1f %5d %8.2f %8.2f %9.1fx\n', th, dec, D, g, dec/max(g,1));
    end
end

fprintf('\n--- 4. E_eff vs the analytic W/g ------------------------------\n');
th = 0.6; r = 12;
fprintf('%-34s %6s %8s %8s\n','scheme','K','E_eff','W/g');
for s = {{'narrowband, full',1,[]}, {'narrowband, every 4th',4,[]}, ...
         {'wideband B/fc=0.05, every 4th',4,beta}}
    nm = s{1}{1}; dec = s{1}{2}; b = s{1}{3};
    phis = a.grid(1:dec:end);
    if isempty(b), fmm = a.fc; Mm = 1; else, fmm = L.subc(a,b,M); Mm = M; end
    bb = beta; if isempty(b), bb = 0; end
    fprintf('%-34s %6d %8.2f %8.2f\n', nm, numel(phis), ...
            L.eeff(a, th, r, fmm, phis), ...
            L.width(a,th,r)/max(L.gap(a,bb,th,dec,Mm),1e-9));
end
