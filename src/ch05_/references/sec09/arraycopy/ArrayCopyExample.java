package ch05_.references.sec09.arraycopy;

public class ArrayCopyExample {
	public static void main(String[] args) {
		//길이 3인 배열
		String[] oldStrArray = { "java", "array", "copy" };
		//길이 5인 배열을 새로 생성
		String[] newStrArray = new String[5];
		//배열 항목 복사 
		// 0 :starting position in the source array. 
		// newStrArray : destination array
		// oldStrArray.length : number of elements to be copied.
		// 두번 쨰 0 : the starting index in the destination array where elements are to be copied.
		/*public static native void arraycopy(Object src,  int  srcPos,
                                        Object dest, int destPos,
                                        int length);*/
		
		System.arraycopy( oldStrArray, 0, newStrArray, 0, oldStrArray.length);
		//배열 항목 출력
		for(int i=0; i<newStrArray.length; i++) {
			System.out.print(newStrArray[i] + ", ");
		}
	}
}
// java, array, copy, null, null, 