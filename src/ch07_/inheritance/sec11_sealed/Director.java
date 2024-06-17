package ch07_.inheritance.sec11_sealed;

//sealed에서 지정된 클래스를 상속하는 것은 가능
public class Director extends Manager {
	@Override
	public void work() {
		System.out.println("제품을 기획합니다.");
	}
}