#ifndef _MATRIX_H_
#define _MATRIX_H_
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef ABS
#define ABS(A) ((A) < 0 ? -(A) : (A))
#endif
template<typename scalar>
__device__ __host__ bool solve3(scalar A[9], scalar b[3], scalar eps = 1e-6) {
	unsigned char pivot = 0, rank = 3, permute[3] = {0,1,2};
	bool	valid = true;
	scalar	t = 0;
	if(ABS(A[0])    < ABS(A[1])) pivot = 1;
	if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
	if(ABS(A[pivot])<= eps) {
		t = A[0]; A[0] = A[6]; A[6] = t;
		t = A[1]; A[1] = A[7]; A[7] = t;
		t = A[2]; A[2] = A[8]; A[8] = t;
		permute[--rank] = 0; pivot = 0;
		if(ABS(A[0])    < ABS(A[1])) pivot = 1;
		if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
		if(ABS(A[pivot])<= eps) {
			t = A[0]; A[0] = A[3]; A[3] = t;
			t = A[1]; A[1] = A[4]; A[4] = t;
			t = A[2]; A[2] = A[5]; A[5] = t;
			permute[--rank] = 0; pivot = 0;
			if(ABS(A[0])    < ABS(A[1])) pivot = 1;
			if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
			if(ABS(A[pivot])<= eps)
				permute[--rank] = 0;
		}
	}
	if(rank > 0) {
		if(pivot == 1) {
			A[0] /= A[1];
			t = A[4]; A[4] = A[3] - A[0]*t; A[3] = t;
			t = A[7]; A[7] = A[6] - A[0]*t; A[6] = t;
			t = b[1]; b[1] = b[0] - A[0]*t; b[0] = t;
			A[0] = A[1]; pivot = 0;
		} else {
			A[1] /= A[pivot];
			A[4] = A[4] - A[1]*A[pivot+3];
			A[7] = A[7] - A[1]*A[pivot+6];
			b[1] = b[1] - A[1]*b[pivot];
		}
		if(pivot == 2) {
			A[0] /= A[2];
			t = A[5]; A[5] = A[3] - A[0]*t; A[3] = t;
			t = A[8]; A[8] = A[6] - A[0]*t; A[6] = t;
			t = b[2]; b[2] = b[0] - A[0]*t; b[0] = t;
			A[0] = A[2];
		} else {
			A[2] /= A[pivot];
			A[5] = A[5] - A[2]*A[pivot+3];
			A[8] = A[8] - A[2]*A[pivot+6];
			b[2] = b[2] - A[2]*b[pivot];
		}
		if(rank > 1) {
			pivot = (ABS(A[4]) < ABS(A[5]) ? 2 : 1);
			if(ABS(A[pivot]) <= eps) {
				if(rank > 2) {
					t = A[3]; A[3] = A[6]; A[6] = t;
					t = A[4]; A[4] = A[7]; A[7] = t;
					t = A[5]; A[5] = A[8]; A[8] = t;
					permute[--rank] = 1;
					pivot = (ABS(A[4]) < ABS(A[5]) ? 2 : 1);
					if(ABS(A[pivot]) <= eps)
						permute[--rank] = 1;
				} else	permute[--rank] = 1;
			}
		}
		if(rank > 1) {
			if(pivot == 2) {
				A[4] /= A[5];
				t = A[8]; A[8] = A[7] - A[4]*t; A[7] = t;
				t = b[2]; b[2] = b[1] - A[4]*t; b[1] = t;
				A[4] = A[5];
			} else {
				A[5] /= A[4];
				A[8] = A[8] - A[5]*A[7];
				b[2] = b[2] - A[5]*b[1];
			}
			if(rank >= 3 && ABS(A[8]) <= eps) permute[--rank] = 2;
		}
	}
	if(rank >= 3) {
		b[2] = b[2] / A[8];
	} else if(ABS(b[2]) > eps) {
		valid = false;
	}
	if(rank >= 2) {
		b[1] = (b[1] - A[7]*b[2]) / A[4];
	} else if(ABS(b[1]) > eps) {
		valid = false;
	}
	if(rank >= 1) {
		b[0] = (b[0] - A[6]*b[2] - A[3]*b[1]) / A[0];
	} else if(ABS(b[0]) > eps) {
		valid = false;
	}
	if(rank <= 1 && permute[1] != 1) {
		t = b[1]; b[1] = b[permute[1]]; b[permute[1]] = t;
	}
	if(rank <= 2 && permute[2] != 2) {
		t = b[2]; b[2] = b[permute[2]]; b[permute[2]] = t;
	}
	return valid;
}
template<typename scalar>
__device__ __host__ bool solve4(scalar A[16], scalar b[4], scalar eps = 1e-6) {
	unsigned char pivot = 0, rank = 4, permute[4] = {0,1,2,3};
	bool	valid = true;
	scalar	t = 0;
	if(ABS(A[0])    < ABS(A[1])) pivot = 1;
	if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
	if(ABS(A[pivot])< ABS(A[3])) pivot = 3;
	if(ABS(A[pivot])<= eps) {
		t = A[0]; A[0] = A[12]; A[12] = t;
		t = A[1]; A[1] = A[13]; A[13] = t;
		t = A[2]; A[2] = A[14]; A[14] = t;
		t = A[3]; A[3] = A[15]; A[15] = t;
		permute[--rank] = 0; pivot = 0;
		if(ABS(A[0])    < ABS(A[1])) pivot = 1;
		if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
		if(ABS(A[pivot])< ABS(A[3])) pivot = 3;
		if(ABS(A[pivot])<= eps) {
			t = A[0]; A[0] = A[8]; A[8] = t;
			t = A[1]; A[1] = A[9]; A[9] = t;
			t = A[2]; A[2] = A[10];A[10]= t;
			t = A[3]; A[3] = A[11];A[11]= t;
			permute[--rank] = 0; pivot = 0;
			if(ABS(A[0])    < ABS(A[1])) pivot = 1;
			if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
			if(ABS(A[pivot])< ABS(A[3])) pivot = 3;
			if(ABS(A[pivot])<= eps) {
				t = A[0]; A[0] = A[4]; A[4] = t;
				t = A[1]; A[1] = A[5]; A[5] = t;
				t = A[2]; A[2] = A[6]; A[6] = t;
				t = A[3]; A[3] = A[7]; A[7] = t;
				permute[--rank] = 0; pivot = 0;
				if(ABS(A[0])    < ABS(A[1])) pivot = 1;
				if(ABS(A[pivot])< ABS(A[2])) pivot = 2;
				if(ABS(A[pivot])< ABS(A[3])) pivot = 3;
				if(ABS(A[pivot])<= eps)
					permute[--rank] = 0;
			}
		}
	}
	if(rank > 0) {
		if(pivot == 1) {
			A[0] /= A[1];
			t = A[5]; A[5] = A[4] - A[0]*t; A[4] = t;
			t = A[9]; A[9] = A[8] - A[0]*t; A[8] = t;
			t = A[13];A[13]= A[12]- A[0]*t; A[12]= t;
			t = b[1]; b[1] = b[0] - A[0]*t; b[0] = t;
			A[0] = A[1]; pivot = 0;
		} else {
			A[1] /= A[pivot];
			A[5] = A[5] - A[1]*A[pivot+4];
			A[9] = A[9] - A[1]*A[pivot+8];
			A[13]= A[13]- A[1]*A[pivot+12];
			b[1] = b[1] - A[1]*b[pivot];
		}
		if(pivot == 2) {
			A[0] /= A[2];
			t = A[6]; A[6] = A[4] - A[0]*t; A[4] = t;
			t = A[10];A[10]= A[8] - A[0]*t; A[8] = t;
			t = A[14];A[14]= A[12]- A[0]*t; A[12]= t;
			t = b[2]; b[2] = b[0] - A[0]*t; b[0] = t;
			A[0] = A[2]; pivot = 0;
		} else {
			A[2] /= A[pivot];
			A[6] = A[6] - A[2]*A[pivot+4];
			A[10]= A[10]- A[2]*A[pivot+8];
			A[14]= A[14]- A[2]*A[pivot+12];
			b[2] = b[2] - A[2]*b[pivot];
		}
		if(pivot == 3) {
			A[0] /= A[3];
			t = A[7]; A[7] = A[4] - A[0]*t; A[4] = t;
			t = A[11];A[11]= A[8] - A[0]*t; A[8] = t;
			t = A[15];A[15]= A[12]- A[0]*t; A[12]= t;
			t = b[3]; b[3] = b[0] - A[0]*t; b[0] = t;
			A[0] = A[3];
		} else {
			A[3] /= A[pivot];
			A[7] = A[7] - A[3]*A[pivot+4];
			A[11]= A[11]- A[3]*A[pivot+8];
			A[15]= A[15]- A[3]*A[pivot+12];
			b[3] = b[3] - A[3]*b[pivot];
		}
	}
	if(rank > 1) {
		pivot = 1;
		if(ABS(A[5])      < ABS(A[6])) pivot = 2;
		if(ABS(A[pivot+4])< ABS(A[7])) pivot = 3;
		if(ABS(A[pivot+4]) <= eps) {
			if(rank > 2) {
				t = A[4]; A[4] = A[rank*4-4]; A[rank*4-4] = t;
				t = A[5]; A[5] = A[rank*4-3]; A[rank*4-3] = t;
				t = A[6]; A[6] = A[rank*4-2]; A[rank*4-2] = t;
				t = A[7]; A[7] = A[rank*4-1]; A[rank*4-1] = t;
				permute[--rank] = 1; pivot = 1;
				if(ABS(A[5])      < ABS(A[6])) pivot = 2;
				if(ABS(A[pivot+4])< ABS(A[7])) pivot = 3;
				if(ABS(A[pivot+4])<= eps) {
					if(rank > 2) {
						t = A[4]; A[4] = A[rank*4-4]; A[rank*4-4] = t;
						t = A[5]; A[5] = A[rank*4-3]; A[rank*4-3] = t;
						t = A[6]; A[6] = A[rank*4-2]; A[rank*4-2] = t;
						t = A[7]; A[7] = A[rank*4-1]; A[rank*4-1] = t;
						permute[--rank] = 1; pivot = 1;
						if(ABS(A[5])      < ABS(A[6])) pivot = 2;
						if(ABS(A[pivot+4])< ABS(A[7])) pivot = 3;
					} else	permute[--rank] = 1;
				}
			} else	permute[--rank] = 1;
		}
	}
	if(rank > 1) {
		if(pivot == 2) {
			A[5] /= A[6];
			t = A[10];A[10]= A[9] - A[5]*t; A[9] = t;
			t = A[14];A[14]= A[13]- A[5]*t; A[13]= t;
			t = b[2]; b[2] = b[1] - A[5]*t; b[1] = t;
			A[5] = A[6]; pivot = 1;
		} else {
			A[6] /= A[pivot+4];
			A[10]= A[10]- A[6]*A[pivot+8];
			A[14]= A[14]- A[6]*A[pivot+12];
			b[2] = b[2] - A[6]*b[pivot];
		}
		if(pivot == 3) {
			A[5] /= A[7];
			t = A[11];A[11]= A[9] - A[5]*t; A[9] = t;
			t = A[15];A[15]= A[13]- A[5]*t; A[13]= t;
			t = b[3]; b[3] = b[1] - A[5]*t; b[1] = t;
			A[5] = A[7];
		} else {
			A[7] /= A[pivot+4];
			A[11]= A[11]- A[7]*A[pivot+8];
			A[15]= A[15]- A[7]*A[pivot+12];
			b[3] = b[3] - A[7]*b[pivot];
		}
	}
	if(rank > 2) {
		pivot = (ABS(A[10]) < ABS(A[11]) ? 3 : 2);
		if(ABS(A[pivot+8]) <= eps) {
			if(rank > 3) {
				t = A[8]; A[8] = A[12]; A[12] = t;
				t = A[9]; A[9] = A[13]; A[13] = t;
				t = A[10];A[10]= A[14]; A[14] = t;
				t = A[11];A[11]= A[15]; A[15] = t;
				permute[--rank] = 2;
				pivot = (ABS(A[10]) < ABS(A[11]) ? 3 : 2);
				if(ABS(A[pivot+8])<= eps) {
					if(rank > 3) {
						t = A[8]; A[8] = A[12]; A[12] = t;
						t = A[9]; A[9] = A[13]; A[13] = t;
						t = A[10];A[10]= A[14]; A[14] = t;
						t = A[11];A[11]= A[15]; A[15] = t;
						permute[--rank] = 2;
						pivot = (ABS(A[10]) < ABS(A[11]) ? 3 : 2);
					} else	permute[--rank] = 2;
				}
			} else	permute[--rank] = 2;
		}
	}
	if(rank > 2) {
		if(pivot == 3) {
			A[10] /= A[11];
			t = A[15];A[15]= A[14]- A[10]*t; A[14]= t;
			t = b[3]; b[3] = b[2] - A[10]*t; b[2] = t;
			A[10] = A[11];
		} else {
			A[11] /= A[pivot+8];
			A[15]= A[15]- A[11]*A[pivot+12];
			b[3] = b[3] - A[11]*b[pivot];
		}
		if(rank > 3 && ABS(A[15]) <= eps) permute[--rank] = 3;
	}
	if(rank >= 4) {
		b[3] = b[3] / A[15];
	} else if(ABS(b[3]) > eps) {
		valid = false;
	}
	if(rank >= 3) {
		b[2] = (b[2] - A[14]*b[3]) / A[10];
	} else if(ABS(b[1]) > eps) {
		valid = false;
	}
	if(rank >= 2) {
		b[1] = (b[1] - A[9]*b[2] - A[13]*b[3]) / A[5];
	} else if(ABS(b[1]) > eps) {
		valid = false;
	}
	if(rank >= 1) {
		b[0] = (b[0] - A[4]*b[1] - A[8]*b[2] - A[12]*b[3]) / A[0];
	} else if(ABS(b[0]) > eps) {
		valid = false;
	}
	if(rank <= 1 && permute[1] != 1) {
		t = b[1]; b[1] = b[permute[1]]; b[permute[1]] = t;
	}
	if(rank <= 2 && permute[2] != 2) {
		t = b[2]; b[2] = b[permute[2]]; b[permute[2]] = t;
	}
	if(rank <= 3 && permute[3] != 3) {
		t = b[3]; b[3] = b[permute[3]]; b[permute[3]] = t;
	}
	return valid;
}
#endif
