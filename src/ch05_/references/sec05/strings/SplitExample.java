package ch05_.references.sec05.strings;

public class SplitExample {
	public static void main(String[] args) {
		String board = "1,자바 학습,참조 타입 String을 학습합니다.,홍길동";

		// 문자열을 콤마(,)를 기준으로 분리 
		String[] tokens = board.split(","); // 그 결과를 token 배열에 저장 
		// board 문자열은 콤마로 구분된 여러 정보를 포함
		
		//각 배열 요소를 인덱스를 통해 접근하고 출력
		System.out.println("번호: " + tokens[0]);
		System.out.println("제목: " + tokens[1]);
		System.out.println("내용: " + tokens[2]);
		System.out.println("성명: " + tokens[3]);
		System.out.println();
			
		//for 문을 이용한 읽기
		// length라는 속성으로 길이 파악( 배열의 길이 만큼 반복하여 각 요소 추출) 
		for(int i=0; i<tokens.length; i++) {
			System.out.println(tokens[i]);
		}
	}
}
