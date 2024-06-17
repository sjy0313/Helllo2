package ch11_exception.sec02.exam01;
// try - catch - finally 
// 실행예외(runtime exception) : compiler 가 예외 처리 코드 여부를 검사하지 않는 예외
public class ExceptionHandlingExample2 {
	public static void printLength(String data) {
		try {
			int result = data.length();
			System.out.println("문자 수: " + result);
		} catch(NullPointerException e) { // 상위상속구조 extends exception
			System.out.println(e.getMessage()); //①
			//System.out.println(e.toString()); //②
			//e.printStackTrace(); //③
		} finally {
			System.out.println("[마무리 실행]\n");
		}
	}

	public static void main(String[] args) {
		System.out.println("[프로그램 시작]\n");
		printLength("ThisIsJava");
		printLength(null);
		System.out.println("[프로그램 종료]");
	}
}