package ch02_.variable_type.sec06_String;
/*
 * 문자열 : 참조형
 *  자바에서 문자열 타입은 참조형인데 기본 타입처럼 사용 
 *  참조타입 : S(대문자)tring
 
 *  \t : 2자리 띄어쓰기
 */
public class StringExample {
	public static void main(String[] args) {
		String name = "홍길동";
		String job = "프로그래머";
		System.out.println(name);
		System.out.println(job);

		String str = "나는 \"자바\"를 배웁니다.";
		 //  \ 처리를 통해 " 문자열로 처리.
		System.out.println(str);
		
		System.out.println("12345678901234567890");
		str = "1234\t5678\t90123";
		System.out.println(str);
		
		str = "번호\t이름\t직업 ";
		System.out.println(str);
		
		System.out.print("나는\n"); // 줄바꿈 : line feed 
		System.out.print("자바를\n"); // 줄바꿈 : line feed 
		System.out.print("배웁니다.");
	}
}