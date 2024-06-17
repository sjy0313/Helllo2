package ch06_.classes.sec11.finals.exam01.finalfields;
/*
 * final field and constant(상수: 변하지 않는 값) : 
 * - 변경불가
 * - 읽기전용 : 값을 바꿀 수 없다
 * - 필드 연산자에 초깃값 지정 
 * - 선언을 할 떄 초깃값을 지정하지 않으면 생성자에서 반드시 초깃값을 지정해야함
 */
public class Korean {
	//인스턴스 final 필드 선언
	final String nation = "대한민국";
	final String ssn;
	
	//인스턴스 필드 선언
	String name;
	
	//생성자 선언
	public Korean(String name) {
		// The blank final field ssn may not have been initialized
		this.name = name;
		this.snn = ""; // 빈값이라도 선언해주어야함.
		
	}

	//생성자 선언
	public Korean(String ssn, String name) {
		this.ssn = ssn;
		this.name = name;
	}
}