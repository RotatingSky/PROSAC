#pragma once

#include <opencv2/opencv.hpp>

#define	CV_PROSAC	1


namespace cv
{
	int nchoosek(int n, int k)
	{
		//计算组合数数值
		int ret = n;
		int m = n - 1;
		int d = 2;
		k = MIN(k, n - k);
		while (d <= k)
		{
			ret = ret * m / d;
			m--;
			d++;
		}
		return ret;
	}

	int nStarUpdate(const CvMat* _mask, int count, double beta = 0.9)
	{
		beta = MAX(beta, 0.);
		beta = MIN(beta, 1.);

		uchar* mask = _mask->data.ptr;
		int maxId = -1;		//临时n_star
		int numInners = 0;	//n_star中类内点个数
		for (int i = 0; i < count; i++)
		{
			if (mask[i])
			{
				maxId = i;
				numInners++;
			}
		}

		//计算In_min
		//	0.95 -- 1.64
		//	0.99 -- 2.33
		//	0.995 -- 2.57
		int I_n_star_min = maxId > 0 ? int(2.5 * sqrt(maxId * beta* (1 - beta)) + maxId * beta) : 0;
		if (numInners >= I_n_star_min)
		{
			return maxId;
		}
		else
		{
			return count;
		}
	}

	class ModelEstimator
	{
	public:
		ModelEstimator(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions);
		virtual ~ModelEstimator();

		virtual int runKernel(const CvMat* m1, const CvMat* m2, CvMat* model) = 0;
		virtual bool runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,
			CvMat* mask, double threshold, double confidence = 0.99, int maxIters = 2000);
		virtual bool runPROSAC(CvMat* om1, CvMat* om2, CvMat* model,
			CvMat* mask, double threshold, double confidence = 0.99, int maxIters = 2000,
			int PR_T_N = 1000);
		virtual bool refine(const CvMat*, const CvMat*, const CvMat*, int) { return true; }

	protected:
		virtual void computeReprojError(const CvMat* m1, const CvMat* m2,
			const CvMat* model, CvMat* error) = 0;
		virtual int findInliers(const CvMat* m1, const CvMat* m2,
			const CvMat* model, CvMat* error, CvMat* mask, double threshold);
		void sortPoints(CvMat* m1, CvMat* m2,
			const CvMat* model, CvMat* _mask);
		virtual bool getSubset(const CvMat* m1, const CvMat* m2,
			CvMat* ms1, CvMat* ms2, int maxAttempts = 1000);
		virtual bool getSubsetProsac(CvMat* om1, CvMat* om2,
			CvMat* ms1, CvMat* ms2, int maxAttempts = 1000);
		virtual bool checkSubset(const CvMat* ms1, int count);

		CvRNG rng;
		int modelPoints;
		CvSize modelSize;
		int maxBasicSolutions;
		bool checkPartialSubsets;

		int PR_t;
		int PR_n;
		int PR_n_star;
		double PR_T_n, PR_T_n_1;	//Tn and Tn-1
		double PR_T_n_ratio;
		int _PR_T_n, _PR_T_n_1;		//T'n and T'n-1
	};

	ModelEstimator::ModelEstimator(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions)
	{
		modelPoints = _modelPoints;
		modelSize = _modelSize;
		maxBasicSolutions = _maxBasicSolutions;
		checkPartialSubsets = true;
		rng = cvRNG(-1);
	}

	ModelEstimator::~ModelEstimator()
	{
	}

	bool ModelEstimator::runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,
		CvMat* mask0, double reprojThreshold, double confidence, int maxIters)
	{
		bool result = false;
		cv::Ptr<CvMat> mask = cvCloneMat(mask0);   //标记矩阵，标记内点和外点
		cv::Ptr<CvMat> models, err, tmask;
		cv::Ptr<CvMat> ms1, ms2;

		int iter, niters = maxIters;   //这是迭代次数，默认最大的迭代次数为2000次
		int count = m1->rows*m1->cols, maxGoodCount = 0;
		CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );
		
		if (count < modelPoints)	//使用RANSAC算法时，modelPoints为4
			return false;

		models = cvCreateMat(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1);
		err = cvCreateMat(1, count, CV_32FC1);
		tmask = cvCreateMat(1, count, CV_8UC1);

		if (count > modelPoints)	//多于4个点
		{
			ms1 = cvCreateMat(1, modelPoints, m1->type);
			ms2 = cvCreateMat(1, modelPoints, m2->type);
		}
		else
		{
			niters = 1;
			ms1 = cvCloneMat(m1);
			ms2 = cvCloneMat(m2);
		}

		for (iter = 0; iter < niters; iter++)
		{
			int i, goodCount, nmodels;
			if (count > modelPoints)
			{
				//调用这个函数，300为循环次数，就是从序列中随机选取4组点
				bool found = getSubset(m1, m2, ms1, ms2, 300);
				if (!found)
				{
					if (iter == 0)
						return false;
					break;
				}
			}

			nmodels = runKernel(ms1, ms2, models);	//通过给定的4组点计算出矩阵

			if (nmodels <= 0)
				continue;
			for (i = 0; i < nmodels; i++)
			{
				CvMat model_i;
				cvGetRows(models, &model_i, i*modelSize.height, (i + 1)*modelSize.height);
				goodCount = findInliers(m1, m2, &model_i, err, tmask, reprojThreshold);
				//当前内点集元素数大于最优内点集元素数
				if (goodCount > MAX(maxGoodCount, modelPoints - 1))
				{
					std::swap(tmask, mask);		//交换mask
					cvCopy(&model_i, model);	//更新最优模型
					maxGoodCount = goodCount;
					niters = cvRANSACUpdateNumIters(confidence,
						(double)(count - goodCount) / count, modelPoints, niters);
				}
			}
		}

		if (maxGoodCount > 0)
		{
			if (mask != mask0)
				cvCopy(mask, mask0);
			return true;
		}

		return result;
	}

	bool ModelEstimator::runPROSAC(CvMat* om1, CvMat* om2, CvMat* model,
		CvMat* mask0, double reprojThreshold, double confidence, int maxIters, int PR_T_N)
	{
		bool result = false;
		cv::Ptr<CvMat> mask = cvCloneMat(mask0);   //标记矩阵，标记内点和外点
		cv::Ptr<CvMat> models, err, tmask;
		cv::Ptr<CvMat> ms1, ms2;

		int iter, niters = maxIters;   //这是迭代次数，默认最大的迭代次数为2000次
		int count = om1->rows*om1->cols, maxGoodCount = 0;
		int maxGoodIter = -1, maxGoodInternal = -1;
		CV_Assert(CV_ARE_SIZES_EQ(om1, om2) && CV_ARE_SIZES_EQ(om1, mask));

		if (count < modelPoints)	//使用RANSAC算法时，modelPoints为4
			return false;

		models = cvCreateMat(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1);
		err = cvCreateMat(1, count, CV_32FC1);
		tmask = cvCreateMat(1, count, CV_8UC1);

		if (count > modelPoints)	//多于4个点
		{
			ms1 = cvCreateMat(1, modelPoints, om1->type);
			ms2 = cvCreateMat(1, modelPoints, om2->type);
		}
		else
		{
			niters = 1;
			ms1 = cvCloneMat(om1);
			ms2 = cvCloneMat(om2);
		}

		//0.初始化变量
		PR_t = 0;
		PR_n = modelPoints - 1;
		PR_n_star = count;
		_PR_T_n = 1;	//T'm = 1
		PR_T_n_ratio = ((double)PR_T_N) / (double)nchoosek(count, modelPoints);
		PR_T_n = PR_T_n_ratio * (double)nchoosek(PR_n, modelPoints);

		for (iter = 0; iter < niters; iter++)
		{
			int i, goodCount, nmodels;
			if (count > modelPoints)
			{
				//1.&2.选择生成集并半随机采样
				bool found = getSubsetProsac(om1, om2, ms1, ms2, 300);
				if (!found)
				{
					if (iter == 0)
						return false;
					break;
				}
			}

			//3.计算模型
			nmodels = runKernel(ms1, ms2, models);
			if (nmodels <= 0)
				continue;

			//4.模型验证
			for (i = 0; i < nmodels; i++)
			{
				//4.1计算支撑点集
				CvMat model_i;
				cvGetRows(models, &model_i, i*modelSize.height, (i + 1)*modelSize.height);
				goodCount = findInliers(om1, om2, &model_i, err, tmask, reprojThreshold);
				//4.2更新模型和迭代参数
				if (goodCount > MAX(maxGoodCount, modelPoints - 1))
				{
					std::swap(tmask, mask);		//交换mask
					cvCopy(&model_i, model);	//更新最优模型
					maxGoodCount = goodCount;
					//更新PR_n_star
					PR_n_star = nStarUpdate(mask, count, 0.9);
					//更新迭代次数
					niters = cvRANSACUpdateNumIters(confidence,
						(double)(PR_n_star - goodCount) / PR_n_star, modelPoints, niters);
				}
				//4.3样本点重排序
				sortPoints(om1, om2, model, mask);
			}
		}

		if (maxGoodCount > 0)
		{
			if (mask != mask0)
				cvCopy(mask, mask0);
			return true;
		}

		return result;
	}

	int ModelEstimator::findInliers(const CvMat* m1, const CvMat* m2,
		const CvMat* model, CvMat* _err, CvMat* _mask, double threshold)
	{
		int i, count = _err->rows*_err->cols, goodCount = 0;
		const float* err = _err->data.fl;
		uchar* mask = _mask->data.ptr;

		computeReprojError(m1, m2, model, _err);  //计算每组点的投影误差
		threshold *= threshold;
		for (i = 0; i < count; i++)
			goodCount += mask[i] = err[i] <= threshold;	//误差在限定范围内，加入‘内点集’
		return goodCount;
	}

	void ModelEstimator::sortPoints(CvMat* om1, CvMat* om2,
		const CvMat* model, CvMat* _mask)
	{
		int count = om1->rows*om1->cols;
		cv::Ptr<CvMat> _err = cvCreateMat(1, count, CV_32FC1);
		float* err = _err->data.fl;
		int type = CV_MAT_TYPE(om1->type), elemSize = CV_ELEM_SIZE(type);
		int *m1ptr = om1->data.i, *m2ptr = om2->data.i;
		uchar* mask = _mask->data.ptr;

		assert(CV_IS_MAT_CONT(om1->type & om2->type) && (elemSize % sizeof(int) == 0));
		elemSize /= sizeof(int);

		//按照当前的模型进行排序
		computeReprojError(om1, om2, model, _err);
		for (int i = 0; i < count - 1; i++)
		{
			int index = i;
			for (int j = i + 1; j < count; j++)
			{
				if (err[j] < err[index])
				{
					index = j;
				}
			}
			if (index == i)
				continue;
			else
			{
				//交换误差值
				float tmpErr;
				tmpErr = err[index];
				err[index] = err[i];
				err[i] = tmpErr;
				//交换数据点
				for (int k = 0; k < elemSize; k++)
				{
					int temp;
					temp = m1ptr[index*elemSize + k];
					m1ptr[index*elemSize + k] = m1ptr[i*elemSize + k];
					m1ptr[i*elemSize + k] = temp;
					temp = m2ptr[index*elemSize + k];
					m2ptr[index*elemSize + k] = m2ptr[i*elemSize + k];
					m2ptr[i*elemSize + k] = temp;
				}
				//交换掩模值
				uchar tmpMask;
				tmpMask = mask[index];
				mask[index] = mask[i];
				mask[i] = tmpMask;
			}
		}
	}

	bool ModelEstimator::getSubset(const CvMat* m1, const CvMat* m2,
		CvMat* ms1, CvMat* ms2, int maxAttempts)
	{
		cv::AutoBuffer<int> _idx(modelPoints);	//modelPoints所需要最少的样本点个数
		int* idx = _idx;
		int i = 0, j, k, idx_i, iters = 0;
		int type = CV_MAT_TYPE(m1->type), elemSize = CV_ELEM_SIZE(type);
		const int *m1ptr = m1->data.i, *m2ptr = m2->data.i;
		int *ms1ptr = ms1->data.i, *ms2ptr = ms2->data.i;
		int count = m1->cols*m1->rows;

		assert(CV_IS_MAT_CONT(m1->type & m2->type) && (elemSize % sizeof(int) == 0));
		elemSize /= sizeof(int);	//每个数据占用的字节数

		for (; iters < maxAttempts; iters++)
		{
			for (i = 0; i < modelPoints && iters < maxAttempts;)
			{
				idx[i] = idx_i = cvRandInt(&rng) % count;	//随机选取1组点
				for (j = 0; j < i; j++)	//检测是否重复选择
					if (idx_i == idx[j])
						break;
				if (j < i)
					continue;	//重新选择
				for (k = 0; k < elemSize; k++)	//复制点数据
				{
					ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
					ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
				}
				//检测点之间是否共线
				if (checkPartialSubsets && (!checkSubset(ms1, i + 1) || !checkSubset(ms2, i + 1)))
				{
					iters++;	//若共线则重新选择一组
					continue;
				}
				i++;
			}
			if (!checkPartialSubsets && i == modelPoints &&
				(!checkSubset(ms1, i) || !checkSubset(ms2, i)))
				continue;
			break;
		}

		return i == modelPoints && iters < maxAttempts;
	}

	bool ModelEstimator::getSubsetProsac(CvMat* om1, CvMat* om2,
		CvMat* ms1, CvMat* ms2, int maxAttempts)
	{
		cv::AutoBuffer<int> _idx(modelPoints);	//modelPoints所需要最少的样本点个数
		int* idx = _idx;
		int i = 0, j, k, idx_i, iters = 0;
		int type = CV_MAT_TYPE(om1->type), elemSize = CV_ELEM_SIZE(type);
		const int *m1ptr = om1->data.i, *m2ptr = om2->data.i;
		int *ms1ptr = ms1->data.i, *ms2ptr = ms2->data.i;
		int count = om1->cols*om1->rows;

		assert(CV_IS_MAT_CONT(om1->type & om2->type) && (elemSize % sizeof(int) == 0));
		elemSize /= sizeof(int);	//每个数据占用的字节数

		if ((++PR_t == _PR_T_n) && (PR_n < PR_n_star - 1))
		{
			PR_n++;
			PR_T_n_1 = PR_T_n;
			_PR_T_n_1 = _PR_T_n;
			PR_T_n = PR_T_n_ratio * (double)nchoosek(PR_n, modelPoints);
			_PR_T_n = _PR_T_n_1 + (int)ceil(PR_T_n - PR_T_n_1);
		}

		bool PR_exceeded = false;
		if (_PR_T_n < PR_t)
		{
			PR_exceeded = true;
		}
		int selectNum = PR_exceeded ? modelPoints - 1 : modelPoints;
		int setNum = PR_exceeded ? PR_n - 1 : PR_n;

		for (; iters < maxAttempts; iters++)
		{
			for (i = 0; i < selectNum && iters < maxAttempts;)
			{
				idx[i] = idx_i = cvRandInt(&rng) % setNum;	//随机选取1组点
				for (j = 0; j < i; j++)	//检测是否重复选择
					if (idx_i == idx[j])
						break;
				if (j < i)
					continue;
				for (k = 0; k < elemSize; k++)	//复制点数据
				{
					ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
					ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
				}
				//检测点之间是否共线
				if (checkPartialSubsets && (!checkSubset(ms1, i + 1) || !checkSubset(ms2, i + 1)))
				{
					iters++;	//若共线则重新选择一组
					continue;
				}
				i++;
			}
			if (PR_exceeded)
			{
				//若PR_t超出次数，则强制选择第PR_n个点作为最后一个元素
				idx_i = PR_n;
				for (k = 0; k < elemSize; k++)
				{
					ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
					ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
				}
				i++;
			}
			if (!checkPartialSubsets && i == modelPoints &&
				(!checkSubset(ms1, i) || !checkSubset(ms2, i)))
				continue;
			break;
		}

		return i == modelPoints && iters < maxAttempts;
	}

	bool ModelEstimator::checkSubset(const CvMat* m, int count)
	{
		int j, k, i, i0, i1;
		CvPoint2D64f* ptr = (CvPoint2D64f*)m->data.ptr;

		assert(CV_MAT_TYPE(m->type) == CV_64FC2);

		if (checkPartialSubsets)
			i0 = i1 = count - 1;
		else
			i0 = 0, i1 = count - 1;

		for (i = i0; i <= i1; i++)
		{
			// check that the i-th selected point does not belong
			// to a line connecting some previously selected points
			for (j = 0; j < i; j++)
			{
				double dx1 = ptr[j].x - ptr[i].x;
				double dy1 = ptr[j].y - ptr[i].y;
				for (k = 0; k < j; k++)
				{
					double dx2 = ptr[k].x - ptr[i].x;
					double dy2 = ptr[k].y - ptr[i].y;
					if (fabs(dx2*dy1 - dy2 * dx1) <= FLT_EPSILON * (fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
						break;
				}
				if (k < j)
					break;
			}
			if (j < i)
				break;
		}

		return i >= i1;
	}


	class HomographyEstimator : public ModelEstimator
	{
	public:
		HomographyEstimator(int modelPoints);

		virtual int runKernel(const CvMat* m1, const CvMat* m2, CvMat* model);
		virtual bool refine(const CvMat* m1, const CvMat* m2,
			CvMat* model, int maxIters);

	protected:
		virtual void computeReprojError(const CvMat* m1, const CvMat* m2,
			const CvMat* model, CvMat* error);
	};

	HomographyEstimator::HomographyEstimator(int _modelPoints)
		: ModelEstimator(_modelPoints, cvSize(3, 3), 1)
	{
		assert(_modelPoints == 4 || _modelPoints == 5);
		checkPartialSubsets = false;
	}

	int HomographyEstimator::runKernel(const CvMat* m1, const CvMat* m2, CvMat* H)
	{
		int i, count = m1->rows*m1->cols;
		const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
		const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

		double LtL[9][9], W[9][1], V[9][9];
		CvMat _LtL = cvMat(9, 9, CV_64F, LtL);
		CvMat matW = cvMat(9, 1, CV_64F, W);
		CvMat matV = cvMat(9, 9, CV_64F, V);
		CvMat _H0 = cvMat(3, 3, CV_64F, V[8]);
		CvMat _Htemp = cvMat(3, 3, CV_64F, V[7]);
		CvPoint2D64f cM = { 0,0 }, cm = { 0,0 }, sM = { 0,0 }, sm = { 0,0 };

		for (i = 0; i < count; i++)
		{
			cm.x += m[i].x; cm.y += m[i].y;
			cM.x += M[i].x; cM.y += M[i].y;
		}

		cm.x /= count; cm.y /= count;
		cM.x /= count; cM.y /= count;

		for (i = 0; i < count; i++)
		{
			sm.x += fabs(m[i].x - cm.x);
			sm.y += fabs(m[i].y - cm.y);
			sM.x += fabs(M[i].x - cM.x);
			sM.y += fabs(M[i].y - cM.y);
		}

		if (fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
			fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
			return 0;

		sm.x = count / sm.x; sm.y = count / sm.y;
		sM.x = count / sM.x; sM.y = count / sM.y;

		double invHnorm[9] = { 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };
		double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
		CvMat _invHnorm = cvMat(3, 3, CV_64FC1, invHnorm);
		CvMat _Hnorm2 = cvMat(3, 3, CV_64FC1, Hnorm2);

		cvZero(&_LtL);
		for (i = 0; i < count; i++)
		{
			double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
			double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
			double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
			double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
			int j, k;
			for (j = 0; j < 9; j++)
				for (k = j; k < 9; k++)
					LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
		}
		cvCompleteSymm(&_LtL);

		//cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
		cvEigenVV(&_LtL, &matV, &matW);
		cvMatMul(&_invHnorm, &_H0, &_Htemp);
		cvMatMul(&_Htemp, &_Hnorm2, &_H0);
		cvConvertScale(&_H0, H, 1. / _H0.data.db[8]);

		return 1;
	}

	bool HomographyEstimator::refine(const CvMat* m1, const CvMat* m2,
		CvMat* model, int maxIters)
	{
		CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
		int i, j, k, count = m1->rows*m1->cols;
		const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
		const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
		CvMat modelPart = cvMat(solver.param->rows, solver.param->cols, model->type, model->data.ptr);
		cvCopy(&modelPart, solver.param);

		for (;;)
		{
			const CvMat* _param = 0;
			CvMat *_JtJ = 0, *_JtErr = 0;
			double* _errNorm = 0;

			if (!solver.updateAlt(_param, _JtJ, _JtErr, _errNorm))
				break;

			for (i = 0; i < count; i++)
			{
				const double* h = _param->data.db;
				double Mx = M[i].x, My = M[i].y;
				double ww = h[6] * Mx + h[7] * My + 1.;
				ww = fabs(ww) > DBL_EPSILON ? 1. / ww : 0;
				double _xi = (h[0] * Mx + h[1] * My + h[2])*ww;
				double _yi = (h[3] * Mx + h[4] * My + h[5])*ww;
				double err[] = { _xi - m[i].x, _yi - m[i].y };
				if (_JtJ || _JtErr)
				{
					double J[][8] = 
					{ 
						{ Mx*ww, My*ww, ww, 0, 0, 0, -Mx * ww*_xi, -My * ww*_xi },
						{ 0, 0, 0, Mx*ww, My*ww, ww, -Mx * ww*_yi, -My * ww*_yi }
					};
					
					for (j = 0; j < 8; j++)
					{
						for (k = j; k < 8; k++)
							_JtJ->data.db[j * 8 + k] += J[0][j] * J[0][k] + J[1][j] * J[1][k];
						_JtErr->data.db[j] += J[0][j] * err[0] + J[1][j] * err[1];
					}
				}
				if (_errNorm)
					*_errNorm += err[0] * err[0] + err[1] * err[1];
			}
		}
		
		cvCopy(solver.param, &modelPart);
		return true;
	}

	void HomographyEstimator::computeReprojError(const CvMat* m1, const CvMat* m2,
		const CvMat* model, CvMat* _err)
	{
		int i, count = m1->rows*m1->cols;
		const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
		const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
		const double* H = model->data.db;
		float* err = _err->data.fl;

		for (i = 0; i < count; i++)	//保存每组点的投影误差，对应单映矩阵变换公式
		{
			double ww = 1. / (H[6] * M[i].x + H[7] * M[i].y + 1.);
			double dx = (H[0] * M[i].x + H[1] * M[i].y + H[2])*ww - m[i].x;
			double dy = (H[3] * M[i].x + H[4] * M[i].y + H[5])*ww - m[i].y;
			err[i] = (float)(dx*dx + dy * dy);
		}
	}


	template<typename T> int icvCompressPoints(T* ptr, const uchar* mask, int mstep, int count)
	{
		int i, j;
		for (i = j = 0; i < count; i++)
			if (mask[i*mstep])
			{
				if (i > j)
					ptr[j] = ptr[i];
				j++;
			}
		return j;
	}

	CV_IMPL int cvFindHomographyProsac(const CvMat* objectPoints, const CvMat* imagePoints,
		CvMat* __H, int method, double ransacReprojThreshold, int prosac_T_N, CvMat* mask)
	{
		const double confidence = 0.995;
		const int maxIters = 2000;	//修改这里来修改迭代次数
		const double defaultRANSACReprojThreshold = 3;
		bool result = false;
		Ptr<CvMat> m, M, tempMask;

		double H[9];
		CvMat matH = cvMat(3, 3, CV_64FC1, H);	//这就是单应矩阵，矩阵初始化
		int count;

		CV_Assert(CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints));

		count = MAX(imagePoints->cols, imagePoints->rows);    //序列个数
		CV_Assert(count >= 4);
		if (ransacReprojThreshold <= 0)
			ransacReprojThreshold = defaultRANSACReprojThreshold;

		m = cvCreateMat(1, count, CV_64FC2);
		cvConvertPointsHomogeneous(imagePoints, m);  //转换齐次坐标

		M = cvCreateMat(1, count, CV_64FC2);
		cvConvertPointsHomogeneous(objectPoints, M);

		if (mask)
		{
			CV_Assert(CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
				(mask->rows == 1 || mask->cols == 1) &&
				mask->rows*mask->cols == count);
		}
		if (mask || count > 4)
			tempMask = cvCreateMat(1, count, CV_8U);
		if (!tempMask.empty())
			cvSet(tempMask, cvScalarAll(1.));

		HomographyEstimator estimator(MIN(count, 4));   //参数是一个小于等于4的值，只有大于4，才能用RANSAC计算
		if (count == 4)
			method = 0;
		if (method == CV_PROSAC)
			result = estimator.runPROSAC(M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters, prosac_T_N);
		else if (method == CV_RANSAC)
			result = estimator.runRANSAC(M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
		else
			result = estimator.runKernel(M, m, &matH) > 0;
		if (result && count > 4)
		{
			icvCompressPoints((CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count);  //压缩，使序列紧凑
			count = icvCompressPoints((CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count);
			M->cols = m->cols = count;    //筛选过后，这个count是内点的个数
			if (method == CV_RANSAC || method == CV_PROSAC)
				estimator.runKernel(M, m, &matH);  //重新计算最终的单应矩阵，matH
			estimator.refine(M, m, &matH, 10);
		}
		if (result)
			cvConvert(&matH, __H);
		if (mask && tempMask)
		{
			if (CV_ARE_SIZES_EQ(mask, tempMask))    //复制这个矩阵
				cvCopy(tempMask, mask);
			else
				cvTranspose(tempMask, mask);        //行列调换的 复制这个矩阵
		}
		return (int)result;
	}

	Mat findHomographyProsac(InputArray _points1, InputArray _points2,
		int method, int prosac_T_N, double ransacReprojThreshold = 3, OutputArray _mask = noArray())
	{
		Mat points1 = _points1.getMat(), points2 = _points2.getMat();
		int npoints = points1.checkVector(2);	//返回矩阵的序列个数
		CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
			points1.type() == points2.type());	//检验初始条件是否正确

		Mat H(3, 3, CV_64F);
		CvMat _pt1 = points1, _pt2 = points2;
		CvMat matH = H, c_mask, *p_mask = 0;
		if (_mask.needed())
		{
			_mask.create(npoints, 1, CV_8U, -1, true);
			p_mask = &(c_mask = _mask.getMat());
		}
		bool ok = cvFindHomographyProsac(&_pt1, &_pt2, &matH, method, ransacReprojThreshold, prosac_T_N, p_mask) > 0;	//函数调用
		if (!ok)
			H = Scalar(0);
		return H;
	}
}
