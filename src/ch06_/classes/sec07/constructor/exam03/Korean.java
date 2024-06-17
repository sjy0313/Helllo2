package ch06_.classes.sec07.constructor.exam03;

public class Korean {
	// 필드 선언
	String nation = "대한민국";
	String name;
	String ssn;

	// 생성자 선언
	// this : 인스턴스 변수+메소드 
	// java 에서 this 키워드를 사용하면 인스턴스 변수/메소드나 생성자 내의 로컬 변수를 구분
	// 객체 자체를 참조하여 다양한 용도로 쓰임
	
	//예를 들어, 생성자의 매개변수 이름이 인스턴스 변수 이름과 동일할 때,
	//this 를 사용하여 인스턴스 변수를 참조합니다.
	//현재 객체의 주소값을 반환합니다. 이를 통해 객체 자체를 다른 메소드로 전달하거나 현재 객체에 대한 정보를 출력할 수 있습니다.
	
	// 중요 )
	//같은 클래스의 다른 생성자를 호출할 때 사용됩니다. 이를 '생성자 체이닝(Constructor Chaining)'이라고 합니다.
	
	public Korean(String name, String ssn) {
		System.out.println("Korean : this=" + this);																																													
		this.name = name;
		this.ssn = ssn;
	}
}