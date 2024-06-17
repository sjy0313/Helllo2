package ch05_.references.sec06.arraytype;

public class ArrayCreateByNewExample2 {
	public static void main(String[] args) {
		// 문자열 배열
		// 배열 변수 선언과 배열 생성 : 문자열 배열의 초깃값은 null
		String[] arr3 = new String[3];
		
		for(int i=0; i<3; i++) {
			System.out.print("arr3[" + i + "] : " + arr3[i] + ", ");
		}
		System.out.println();
		//배열 항목의 값 변경
		arr3[0] = "1월";
		arr3[1] = "2월";
		arr3[2] = "3월";
		//배열 항목의 변경값 출력
		for(int i=0; i<3; i++) {
			System.out.print("arr3[" + i + "] : " + arr3[i] + ", ");
		}
	}
}