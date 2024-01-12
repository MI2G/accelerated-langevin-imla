function [A, AT, ATA, H_FFT, HC_FFT] = box_blur(image_sz, sz_kernel, make_column)
%BOX_BLUR create box blur and operators for a box blur of size sz_kernel

%%%% function handle for uniform blur operator (acts on the image
%%%% coefficients)
h = ones(1, sz_kernel);
lh = length(h);
h = h/sum(h);
h = [h zeros(1,image_sz(1)-length(h))];
h = cshift(h,-(lh-1)/2);
h = h'*h;

%%% operators A and A'
H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

%%% operators A and A' operating with column vectors

if make_column
    
    A = @(x) reshape(real(ifft2(H_FFT.*fft2(reshape(x, image_sz)))), [],1); % A operator
    AT = @(x) reshape(real(ifft2(HC_FFT.*fft2((reshape(x, image_sz))))), [],1); % A transpose operator
    ATA = @(x) reshape(real(ifft2((HC_FFT.*H_FFT).*fft2((reshape(x, image_sz))))), [],1); % AtA operator

else

    A = @(x) real(ifft2(H_FFT.*fft2(reshape(x, image_sz)))); % A operator
    AT = @(x) real(ifft2(HC_FFT.*fft2((reshape(x, image_sz))))); % A transpose operator
    ATA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((reshape(x, image_sz))))); % AtA operator
end

end

