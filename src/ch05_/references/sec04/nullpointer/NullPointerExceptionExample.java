package ch05_.references.sec04.nullpointer;

public class NullPointerExceptionExample {
	public static void main(String[] args) {
		int[] intArray = null; // 어디에 저장한지 모름(공간이 할당되지 않은 상태) 즉, 변수만 선언된 상태 
		//intArray[0] = 10; //NullPointerException
		
		
		String str = null; // "null" 문자열 그 자체이므로 개수 4개로 출력
		// Cannot invoke "String.length()" because "str" is null -> str 변수가 null 이므로 invoke(불러옴)할 수 없음
		// System.out.println("총 문자 수: " + str.length() );//NullPointerException
	}
}