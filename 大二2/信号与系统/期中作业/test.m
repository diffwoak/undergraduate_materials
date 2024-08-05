Pathh='Einstein.jpg';
Pathl='Marilyn.jpg';
imh=imread(Pathh);
iml=imread(Pathl);
imgh=rgb2gray(imh);
imgl=rgb2gray(iml);%将彩色图变成二维灰度图图像
subplot(331)
imshow(imgh);
subplot(334)
imshow(imgl);
imgh2=double(imgh);
imgl2=double(imgl);%将数据改成double类型，方便进行数据变换
fh=fft2(imgh2);
fl=fft2(imgl2);
fh2=fftshift(fh);
fl2=fftshift(fl);
subplot(332)
imshow(log(abs(fh2)+1),[]);%直接显示图像，未移动零频分量
title('傅里叶系数')
subplot(335)
imshow(log(abs(fl2)+1),[]);%直接显示图像，移动零频分量
title('傅里叶系数')

D0=15;
[Mh,Nh]=size(fh2);
[Ml,Nl]=size(fl2);
ml=floor(Ml/2);
nl=floor(Nl/2);
mh=floor(Mh/2);
nh=floor(Nh/2);
Hl=zeros(Ml,Nl);
Hh=zeros(Mh,Nh);
for i=1:Ml
    for j=1:Nl
        D=sqrt((i-ml)^2+(j-nl)^2);
        Hl(i,j)=exp(-1/2*D^2/D0^2);
    end
end
for i=1:Mh
    for j=1:Nh
        D=sqrt((i-mh)^2+(j-nh)^2);
        Hh(i,j)=1-exp(-1/2*D^2/D0^2);
    end
end
subplot(333)
fh3=fh2.*Hh;
imshow(log(abs(fh3)+1),[]);
title('高斯高通滤波结果')
subplot(336)
fl3=fl2.*H;
imshow(log(abs(fl3)+1),[])       
title('高斯低通滤波结果')

imgh3 = (ifft2(ifftshift(fh3)));
imgh4 = uint8(real(imgh3)); %real函数表示留下复数的实部
imgl3 = (ifft2(ifftshift(fl3)));
imgl4 = uint8(real(imgl3)); %real函数表示留下复数的实部
imgh4 = imresize(imgh4, [Ml Nl]); 
img=imgh4+imgl4;
subplot(337)
imshow(imgh4);
subplot(338)
imshow(imgl4);
subplot(339)
imshow(img);
imwrite(img, 'result.jpg');