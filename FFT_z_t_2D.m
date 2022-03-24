clc
clear all
close all

np_opt=16
parpool(np_opt)


pathMean{1}='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/meanVal_1/';
pathMean{2}='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/meanVal_2/';
pathMean{3}='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/meanVal_3/';
%pathMean{4}='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/meanVal_4/';

pathIn='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/';
pathOut='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/FFT_Z/';


[n_path,n_subPath]=size(pathMean)


%     if i~=1

if isfile(fullfile(pathMean{1},'input.i3d'))
    NLSt=read_namelist(fullfile(pathMean{1},'input.i3d'));
elseif isfile(fullfile(pathMean{1},'../input.i3d'))
    NLSt=read_namelist(fullfile(pathMean{1},'../input.i3d'));
else
    error('input.i3d is not found')
end
nx=NLSt.BasicParam.nx; ny=NLSt.BasicParam.ny;
nz=NLSt.BasicParam.nz; beta=NLSt.BasicParam.beta;
xlx=NLSt.BasicParam.xlx;nx1=NLSt.InOutParam.ivs;
nx2=NLSt.InOutParam.ive
Re=NLSt.BasicParam.re; nu=1./Re;

%     end
x_0=1./(4.91^2/Re);


x=linspace(0,xlx,nx);y=load([pathMean{1},'/yp.dat']);
z=linspace(1,nz,nz);
[Y,X,Z]=meshgrid(y,x,z);


x=linspace(0,xlx,nx);
nx_snap=nx2-nx1+1;
for I=1:length(pathMean)
    if I ==1
        spanwiseAverage(pathMean{I},nx,ny,nz,"true")
        umean_xy=read_binary([pathMean{I},'2D_umean.dat'],nx,ny,1,1);
        vmean_xy=read_binary([pathMean{I},'2D_vmean.dat'],nx,ny,1,1);
        wmean_xy=read_binary([pathMean{I},'2D_wmean.dat'],nx,ny,1,1);
    else
        if length(pathMean{I})>0
            spanwiseAverage(pathMean{I},nx,ny,nz,"true")
            umean_xy=umean_xy+read_binary([pathMean{I},'2D_umean.dat'],nx,ny,1,1);
            vmean_xy=vmean_xy+read_binary([pathMean{I},'2D_vmean.dat'],nx,ny,1,1);
            wmean_xy=wmean_xy+read_binary([pathMean{I},'2D_wmean.dat'],nx,ny,1,1);
        end
        
    end
end
nt=umean_xy(1,end);
umean_xy= umean_xy/nt;     vmean_xy=vmean_xy/nt;
wmean_xy=wmean_xy/nt;      u_inf=umean_xy(:,end-1);
for ii=1:ny
    umean_xy(:,ii)= umean_xy(:,ii)./u_inf;
    vmean_xy(:,ii)= vmean_xy(:,ii)./u_inf;
    wmean_xy(:,ii)= wmean_xy(:,ii)./u_inf;
end


Umean=repmat(umean_xy,1,1,nz);
Vmean=repmat(vmean_xy,1,1,nz);
Wmean=repmat(wmean_xy,1,1,nz);



fSt=1;
fEnd=3000;
step=1;
Ns=100;
l=0;N=0;Nb=0;


modZ=[1:32];
nxy=nx_snap*ny;

uxn=zeros(nx_snap,ny,length(modZ),Ns);
uyn=zeros(nx_snap,ny,length(modZ),Ns);
uzn=zeros(nx_snap,ny,length(modZ),Ns);

parfor n=fSt:fEnd
    N=(n-fSt+step)/step;
    tic
    checkStop=load('checkStop');
    if checkStop==1
        error('That is enough')
    end
    
    disp(num2str([ n fEnd],'Snaphsot   %i / %i'))
    
    UX=read_binary(fullfile(pathIn,num2str([n],'ux%.5i')),nx_snap,ny,nz,1)-Umean(nx1:nx2,:,:);
    UY=read_binary(fullfile(pathIn,num2str([n],'uy%.5i')),nx_snap,ny,nz,1)-Vmean(nx1:nx2,:,:);
    UZ=read_binary(fullfile(pathIn,num2str([n],'uz%.5i')),nx_snap,ny,nz,1)-Wmean(nx1:nx2,:,:);
    
    %     warning('Mean is not subtracted')
    
    ux_FFT = single(fft(UX,[],3)/nz);
    uy_FFT = single(fft(UY,[],3)/nz);
    uz_FFT = single(fft(UZ,[],3)/nz);
    
    
    for k=modZ
        
        uxn(:,:,k,N)=squeeze(ux_FFT(:,:,k));
        uyn(:,:,k,N)=squeeze(uy_FFT(:,:,k));
        uzn(:,:,k,N)=squeeze(uz_FFT(:,:,k));
        %         save(filename,'ux','uy','uz','-v7.3')
        
    end
    
    if mod(N,Ns)==0
        tic 
        Nb=Nb+1;
%         system(['rm -rf ',fullfile(pathOut,num2str([Nb] ,'Data_%.5i'))]);
        system(['mkdir ',fullfile(pathOut,num2str([Nb] ,'Data_%.5i'))]);
        for k=modZ
            filename=fullfile(pathOut,num2str([Nb, k] ,'Data_%.5i/%.5i.mat'));
            
            SIZE=size(real(squeeze(uxn(:,:,k,:))));
            h5create(filename,'/ux_r',SIZE);
            h5write(filename,'/ux_r',real(squeeze(uxn(:,:,k,:))));
            
            h5create(filename,'/uy_r',SIZE);
            h5write(filename,'/uy_r',real(squeeze(uyn(:,:,k,:))));
            
            h5create(filename,'/uz_r',SIZE);
            h5write(filename,'/uz_r',real(squeeze(uzn(:,:,k,:))));
            
            h5create(filename,'/ux_i',SIZE);
            h5write(filename,'/ux_i',imag(squeeze(uxn(:,:,k,:))));
            
            h5create(filename,'/uy_i',SIZE);
            h5write(filename,'/uy_i',imag(squeeze(uyn(:,:,k,:))));
            
            h5create(filename,'/uz_i',SIZE);
            h5write(filename,'/uz_i',imag(squeeze(uzn(:,:,k,:))));            
            
        end

        T1=toc;
        disp(['Writing time =',num2str(T1)])
    end
    toc
end




