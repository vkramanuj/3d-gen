unsigned int getIndex(const int y,const int x, const int z, const int width, const int height);
double dis(
		__global double *img1,const int y1, const int x1,
		__global double *img2,const int y2, const int x2,
		const int patchWidth, const int patchHeight,
		const int width, const int height
		);
int random (int start, int end,unsigned int seed);
unsigned int getIndex(const int y, const int x, const int z, const int width, const int height)
{
	int myX = clamp(x, 0, width-1);
	int myY = clamp(y, 0, height-1);
	return ((myY*width+myX)*3+z);
}

double dis(
		__global double *img1,const int y1, const int x1,
		__global double *img2,const int y2, const int x2,
		const int patchWidth, const int patchHeight,
		const int width, const int height
		)
{
	double diff=0;
	for (int j=-patchHeight/2;j<=patchHeight/2;++j)
		for (int i=-patchWidth/2;i<=patchWidth/2;++i)
			for (int k=0;k<3;++k)
			{
				double t=(img1[getIndex(j+y1,i+x1,k,width,height)]-img2[getIndex(j+y2,i+x2,k,width,height)]);
				diff+=t*t;
			}
	return diff;

}
int random (int start, int end,unsigned int seed)
{
	unsigned int num=(seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	return num%(end-start+1)+start;
}
#define D(y1,x1,y2,x2) dis(img1,(y1),(x1),img2,y2,x2,patchWidth,patchHeight,width,height)

#define nff(i,j,k) output[((i) * width +(j))*3+(k)]

#define MAXINT 9999999.0
__kernel void randomfill(const int patchHeight, const int patchWidth,
			const int height,const int width,
			__global double *img1,__global double *img2, __global double * output)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int seed=y<<(16+x);
	int ty=seed=nff(y,x,0)=random(0,height,seed);
	int tx=nff(y,x,1)=random(0,width,seed);
	nff(y,x,2)=D(y,x, ty,tx);
}
__kernel void propagate( const int patchHeight, const int patchWidth,
			const int height,const int width, 			const int iteration,
		__global double *img1,__global double *img2,__global double * output)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int direction=1;
	if (iteration%2==0)
	{
		x=width-x-1;
		y=height-y-1;
		direction=-1;
	}

	int dir=direction; //temp direction
    if (direction==-1)
    	if (y+1>=height || x+1>=width) return;

	double currentD,topD,leftD;
	//compute intensitive part
	dir=direction;
	if (nff(y-dir,x,0)+dir>=height || nff(y-dir,x,0)+dir<0)
		topD=MAXINT;
	else
		topD=D(y,x , nff(y-dir,x,0)+dir , nff(y-dir,x,1));

	if (nff(y,x-dir,1)+dir>=width || nff(y,x-dir,1)+dir<0)
		leftD=MAXINT;
	else
		leftD=D(y,x , nff(y,x-dir,0) , nff(y,x-dir,1)+dir);


	dir=direction;
	currentD=nff(y,x,2);

	if (topD<currentD)
	{
		nff(y,x,0)=nff(y-dir,x,0)+dir;
		nff(y,x,1)=nff(y-dir,x,1);
		currentD=nff(y,x,2)=topD;
	}

	if (leftD<currentD)
	{
		nff(y,x,0)=nff(y,x-dir,0);
		nff(y,x,1)=nff(y,x-dir,1)+dir;
		currentD=nff(y,x,2)=leftD;
	}

	//random search
	unsigned int seed=1;
	int w=width,h=height;
    while (h>1 && w>1)
    {
    	int x1,y1,x2,y2;
    	y1=y-h/2;
    	x1=x-w/2;
    	y2=y+h/2;
    	x2=x+w/2;
    	if (x1<0) x1=0;
    	if (y1<0) y1=0;
    	if (x2>=width)
    		x2=width-1;
    	if (y2>=height)
    		y2=height-1;

    	int targetX=seed=random(x1,x2,seed);
    	int targetY=seed=random(y1,y2,seed);

        double newD=D(y,x, targetY,targetX);
        if (newD<nff(y,x,2))
        {
            nff(y,x,0)=targetY;
            nff(y,x,1)=targetX;
            nff(y,x,2)=newD;
        }
        w/=2;
        h/=2;
    }

}
